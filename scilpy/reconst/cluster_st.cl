/*
Cluster short tracks tractogram by running Quickbundles inside each voxel.
*/

#define N_RESAMPLE 5
#define MAX_N_CLUSTERS 10
#define MAX_NB_STRL 0
#define MAX_DEVIATION 0.7f
#define DIST_METRIC 0

uint get_nb_points(uint index, __global const uint* strl_offsets)
{
    return strl_offsets[index + 1] - strl_offsets[index];
}

uint get_nb_streamlines(uint voxel_id, __global const uint* strl_per_vox_offsets)
{
    return strl_per_vox_offsets[voxel_id + 1] - strl_per_vox_offsets[voxel_id];
}

// Not garanteed that each segment is the same length but probably good enough
float4 interp_along_track(__global const float4* track, const int nb_points, float t)
{
    const float t_float_index = (float)(nb_points - 1) * t;
    const int t_int_floor = (int)t_float_index;
    const int t_int_ceil = t_int_floor + 1;

    if(t_int_ceil > nb_points - 1)
        return track[nb_points - 1];
    
    const float4 p_prev = track[t_int_floor];
    const float4 p_next = track[t_int_ceil];

    const float r = t_float_index - (float)t_int_floor;
    return r*p_next + (1.0f - r)*p_prev;
}

void resample_track_to_points(__global const float4* in_track,
                              const int nb_points, float4* out_track)
{
    const float delta_t = 1.0f / (float)(N_RESAMPLE - 1);
    for (int i = 0; i < N_RESAMPLE; i++)
    {
        out_track[i] = interp_along_track(in_track, nb_points, i*delta_t);
    }
}

void resample_track_to_vecs(__global const float4* in_track,
                            const int nb_points, float4* out_dirs)
{
    // when comparing only directions, N_RESAMPLE is the number of directions
    const float delta_t = 1.0f / (float)(N_RESAMPLE);
    float4 prev_point = interp_along_track(in_track, nb_points, 0.0f);
    for (int i = 1; i < N_RESAMPLE + 1; i++)
    {
        const float4 curr_point = interp_along_track(in_track, nb_points,
                                                     i*delta_t);
        const float3 v = normalize(curr_point.xyz - prev_point.xyz);
        out_dirs[i-1] = (float4)(v.x, v.y, v.z, 0.0f);
    }
}

// second version where s and t are arrays of directions
float mad_min(const float4* s_prime, const float4* t_prime,
              bool* min_is_flipped)
{
    float direct = 0.0f;
    float flipped = 0.0f;
    float3 s_dir;
    float3 t_dir;
    float3 t_dir_flipped;
    for(int i = 0; i < N_RESAMPLE; ++i)
    {
        // we need to normalize because centroid isn't normalized
        s_dir = normalize(s_prime[i].xyz);
        t_dir = normalize(t_prime[i].xyz);
        // last direction in reverse
        t_dir_flipped = - normalize(t_prime[N_RESAMPLE-1-i].xyz);
        direct += acos(dot(s_dir, t_dir));
        flipped += acos(dot(s_dir, t_dir_flipped));
    }

    *min_is_flipped = flipped < direct;
    if(*min_is_flipped)
    {
        return flipped / (float)(N_RESAMPLE);
    }

    return direct / (float)(N_RESAMPLE);
}

float mdf_min(const float4* s, const float4* t,
          bool* min_is_flipped)
{
    float direct = 0.0f;
    float flipped = 0.0f;
    for(int i = 0; i < N_RESAMPLE; ++i)
    {
        direct += fast_distance(s[i].xyz, t[i].xyz);
        flipped += fast_distance(s[i].xyz, t[N_RESAMPLE - 1 -i].xyz);
    }

    *min_is_flipped = flipped < direct;
    if(*min_is_flipped)
    {
        return flipped / (float)N_RESAMPLE;
    }
    return direct / (float)N_RESAMPLE;
}

// return the id of the closest cluster
int qb_orientation(__global const float4* track,
                   const int nb_points,
                   const float max_deviation,
                   const int nb_clusters,
                   float4* cluster_track_sums,
                   int* cluster_track_counts)
{
    // 1. resample track
    float4 resampled_dirs[N_RESAMPLE];
    resample_track_to_vecs(track, nb_points, resampled_dirs);

    // 2. compare the resampled track to all cluster centroids
    float min_dist = FLT_MAX;
    int best_cluster_id = 0;
    bool needs_flip = false;
    for(int i = 0; i < nb_clusters; ++i)
    {
        // compute distance to each cluster
        bool _needs_flip;
        const float dist = mad_min(resampled_dirs,
                                   &cluster_track_sums[i*N_RESAMPLE],
                                   &_needs_flip);

        if(dist < min_dist)
        {
            min_dist = dist;
            best_cluster_id = i;
            needs_flip = _needs_flip;
        }
    }

    // 3. if min_dist > the distance threshold, add a new cluster
    if(min_dist > max_deviation && nb_clusters < MAX_N_CLUSTERS)
    {
        best_cluster_id = nb_clusters; // create new cluster
        cluster_track_counts[best_cluster_id] = 1;

        // initialize track sum to 0
        for(int i = 0; i < N_RESAMPLE; ++i)
        {
            cluster_track_sums[best_cluster_id*N_RESAMPLE+i] =
                (float4)(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }
    else // we increment the number of tracks in the best cluster
    {
        ++cluster_track_counts[best_cluster_id];
    }

    // 4. add the resampled track to the cluster track sum
    for (int i = 0; i < N_RESAMPLE; i++)
    {
        if(needs_flip)
        {
            // if a flip is needed we must add the opposite vector to the last position
            cluster_track_sums[(best_cluster_id + 1)*N_RESAMPLE - 1 - i] += -resampled_dirs[i];
        }
        else
        {
            cluster_track_sums[best_cluster_id*N_RESAMPLE+i] += resampled_dirs[i];
        }
    }
    return best_cluster_id;
}

int qb_spatial(__global const float4* track, const int nb_points,
               const float max_deviation, const int nb_clusters,
               float4* cluster_track_sums, int* cluster_track_counts)
{
    // 1. resample track
    float4 resampled_points[N_RESAMPLE];
    resample_track_to_points(track, nb_points, resampled_points);

    // 2. compare the resampled track to all cluster centroids
    float min_dist = FLT_MAX;
    int best_cluster_id = 0;
    bool needs_flip = false;
    for(int i = 0; i < nb_clusters; ++i)
    {
        // compute distance to each cluster
        bool _needs_flip;
        const float dist = mdf_min(
            resampled_points, &cluster_track_sums[i*N_RESAMPLE],
            &_needs_flip);

        if(dist < min_dist)
        {
            min_dist = dist;
            best_cluster_id = i;
            needs_flip = _needs_flip;
        }
    }

    // 3. if min_dist > the distance threshold, add a new cluster
    if(min_dist > max_deviation && nb_clusters < MAX_N_CLUSTERS)
    {
        best_cluster_id = nb_clusters; // create new cluster
        cluster_track_counts[best_cluster_id] = 1;

        // initialize track sum to 0
        for(int i = 0; i < N_RESAMPLE; ++i)
        {
            cluster_track_sums[best_cluster_id*N_RESAMPLE+i] =
                (float4)(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }
    else // we increment the number of tracks in the best cluster
    {
        ++cluster_track_counts[best_cluster_id];
    }

    // 4. add the resampled track to the cluster track sum
    for (int i = 0; i < N_RESAMPLE; i++)
    {
        if(needs_flip)
        {
            // if a flip is needed we must add the opposite vector to the last position
            cluster_track_sums[(best_cluster_id + 1)*N_RESAMPLE - 1 - i] += resampled_points[i];
        }
        else
        {
            cluster_track_sums[best_cluster_id*N_RESAMPLE+i] += resampled_points[i];
        }
    }
    return best_cluster_id;
}

int qb_spatial_merge(const float4* track,
                     const float max_deviation, const int nb_clusters,
                     float4* cluster_track_sums, int* cluster_track_counts)
{
    float min_dist = FLT_MAX;
    int best_cluster_id = 0;
    bool needs_flip = false;
    for(int i = 0; i < nb_clusters; ++i)
    {
        // compute distance to each cluster
        bool _needs_flip;
        const float dist = mdf_min(
            track, &cluster_track_sums[i*N_RESAMPLE],
            &_needs_flip);

        if(dist < min_dist)
        {
            min_dist = dist;
            best_cluster_id = i;
            needs_flip = _needs_flip;
        }
    }

    // if min_dist > the distance threshold, add a new cluster
    if(min_dist > max_deviation && nb_clusters < MAX_N_CLUSTERS)
    {
        best_cluster_id = nb_clusters; // create new cluster
        cluster_track_counts[best_cluster_id] = 1;

        // initialize track sum to 0
        for(int i = 0; i < N_RESAMPLE; ++i)
        {
            cluster_track_sums[best_cluster_id*N_RESAMPLE+i] =
                (float4)(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }
    else // we must increment the number of tracks in the best cluster
    {
        ++cluster_track_counts[best_cluster_id];
    }

    // add the resampled track to the cluster track sum
    for (int i = 0; i < N_RESAMPLE; i++)
    {
        if(needs_flip)
        {
            // if a flip is needed we must add the opposite vector to the last position
            cluster_track_sums[(best_cluster_id + 1)*N_RESAMPLE - 1 - i] += track[i];
        }
        else
        {
            cluster_track_sums[best_cluster_id*N_RESAMPLE+i] += track[i];
        }
    }
    return best_cluster_id;
}

int qb_orientation_merge(const float4* track, const float max_deviation,
                         const int nb_clusters, float4* cluster_track_sums,
                         int* cluster_track_counts)
{
    float min_dist = FLT_MAX;
    int best_cluster_id = 0;
    bool needs_flip = false;
    for(int i = 0; i < nb_clusters; ++i)
    {
        // compute distance to each cluster
        bool _needs_flip;
        const float dist = mad_min(
            track, &cluster_track_sums[i*N_RESAMPLE],
            &_needs_flip);

        if(dist < min_dist)
        {
            min_dist = dist;
            best_cluster_id = i;
            needs_flip = _needs_flip;
        }
    }

    // if min_dist > the distance threshold, add a new cluster
    if(min_dist > max_deviation && nb_clusters < MAX_N_CLUSTERS)
    {
        best_cluster_id = nb_clusters; // create new cluster
        cluster_track_counts[best_cluster_id] = 1;

        // initialize track sum to 0
        for(int i = 0; i < N_RESAMPLE; ++i)
        {
            cluster_track_sums[best_cluster_id*N_RESAMPLE+i] =
                (float4)(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }
    else // we must increment the number of tracks in the best cluster
    {
        ++cluster_track_counts[best_cluster_id];
    }

    // add the resampled track to the cluster track sum
    for (int i = 0; i < N_RESAMPLE; i++)
    {
        if(needs_flip)
        {
            // if a flip is needed we must add the opposite vector to the last position
            cluster_track_sums[(best_cluster_id + 1)*N_RESAMPLE - 1 - i] += -track[i];
        }
        else
        {
            cluster_track_sums[best_cluster_id*N_RESAMPLE+i] += track[i];
        }
    }
    return best_cluster_id;
}

__kernel void cluster_per_voxel(__global const float4* all_points,
                                __global const uint* strl_pts_offsets,
                                __global const uint* strl_per_vox_offsets,
                                __global uint* out_cluster_ids)
{
    const int voxel_id = get_global_id(0);
    const uint first_strl_offset = strl_per_vox_offsets[voxel_id];
    const uint nb_streamlines = get_nb_streamlines(voxel_id, strl_per_vox_offsets);

    // QB clustering data structures
    // Cluster ID of each streamline to cluster
    int strl_cluster_ids[MAX_NB_STRL];

    // Non-normalized track directions of each cluster
    float4 cluster_track_sums[MAX_N_CLUSTERS*N_RESAMPLE];

    // Number of tracks belonging to each cluster
    int cluster_track_counts[MAX_N_CLUSTERS];

    // Number of clusters
    uint nb_clusters = 0;

    // iterate through all streamlines and cluster each of them
    for(uint i = 0; i < nb_streamlines; ++i)
    {
        const uint strl_pts_offset = strl_pts_offsets[first_strl_offset + i];
        const uint strl_nb_points = get_nb_points(first_strl_offset+i, strl_pts_offsets);

        if(DIST_METRIC == 0)
        {
            strl_cluster_ids[i] = qb_orientation(&all_points[strl_pts_offset], strl_nb_points,
                                                 MAX_DEVIATION, nb_clusters, cluster_track_sums,
                                                 cluster_track_counts);
        }
        else if(DIST_METRIC == 1)
        {
            strl_cluster_ids[i] = qb_spatial(&all_points[strl_pts_offset], strl_nb_points,
                                             MAX_DEVIATION, nb_clusters, cluster_track_sums,
                                             cluster_track_counts);
        }

        if(strl_cluster_ids[i] == nb_clusters)
        {
            ++nb_clusters;
        }
    }

    // Merge clusters
    // first, copy centroids
    float4 centroids[MAX_N_CLUSTERS*N_RESAMPLE];
    for(uint i = 0; i < nb_clusters; ++i)
    {
        for(uint point_id = 0; point_id < N_RESAMPLE; ++point_id)
        {
            centroids[i*N_RESAMPLE+point_id] = cluster_track_sums[i*N_RESAMPLE+point_id]
                                             / cluster_track_counts[i];
        }
    }
    int merged_cluster_ids[MAX_N_CLUSTERS];
    float4 merged_clusters_track_sums[MAX_N_CLUSTERS*N_RESAMPLE];
    int merged_clusters_track_counts[MAX_N_CLUSTERS];
    uint nb_merged_clusters = 0;

    for(uint i = 0; i < nb_clusters; ++i)
    {
        if(DIST_METRIC == 0)
        {
            merged_cluster_ids[i] = qb_orientation_merge(&centroids[i*N_RESAMPLE],
                                                         MAX_DEVIATION,
                                                         nb_merged_clusters,
                                                         merged_clusters_track_sums,
                                                         merged_clusters_track_counts);
        }
        else if (DIST_METRIC == 1)
        {
            merged_cluster_ids[i] = qb_spatial_merge(&centroids[i*N_RESAMPLE],
                                                     MAX_DEVIATION,
                                                     nb_merged_clusters,
                                                     merged_clusters_track_sums,
                                                     merged_clusters_track_counts);
        }
        if(merged_cluster_ids[i] == nb_merged_clusters)
        {
            ++nb_merged_clusters;
        }
    }

    // on copie les streamlines cluster ids
    for(uint i = 0; i < nb_streamlines; ++i)
    {
        out_cluster_ids[first_strl_offset+i] = merged_cluster_ids[strl_cluster_ids[i]];
    }
}