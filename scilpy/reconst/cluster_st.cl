/*
Cluster short tracks tractogram by running Quickbundles inside each voxel.
*/
#define N_RESAMPLE 5
#define MAX_N_CLUSTERS 10
#define MAX_NB_STRL 0
#define MAX_DEVIATION 0.7f
#define DIST_METRIC 0
#define MERGE_DIST_RATIO 1.0f
#define MERGE_CLUSTERS 1

enum DistMetrics
{
    MAD_MIN = 0,
    MDF_AVG = 1
};

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
                              const int nb_points, const int nb_resample,
                              float4* out_track)
{
    const float delta_t = 1.0f / (float)(nb_resample - 1);
    for (int i = 0; i < nb_resample; i++)
    {
        out_track[i] = interp_along_track(in_track, nb_points, i*delta_t);
    }
}

void resample_track_to_vecs(__global const float4* in_track,
                            const int nb_points, const int nb_resample,
                            float4* out_dirs)
{
    const float delta_t = 1.0f / (float)(nb_resample);
    float4 prev_point = interp_along_track(in_track, nb_points, 0.0f);
    for (int i = 1; i < nb_resample + 1; i++)
    {
        const float4 curr_point = interp_along_track(in_track, nb_points,
                                                     i*delta_t);
        const float3 v = normalize(curr_point.xyz - prev_point.xyz);
        out_dirs[i-1] = (float4)(v.x, v.y, v.z, 0.0f);
    }
}

// second version where s and t are arrays of directions
float mad_min(const float4* s_prime, const float4* t_prime,
              const int nb_directions, bool* min_is_flipped)
{
    float direct = 0.0f;
    float flipped = 0.0f;
    float3 s_dir;
    float3 t_dir;
    float3 t_dir_flipped;
    for(int i = 0; i < nb_directions; ++i)
    {
        // we need to normalize because centroid isn't normalized
        s_dir = normalize(s_prime[i].xyz);
        t_dir = normalize(t_prime[i].xyz);
        // last direction in reverse
        t_dir_flipped = - normalize(t_prime[nb_directions-1-i].xyz);
        direct += acos(dot(s_dir, t_dir));
        flipped += acos(dot(s_dir, t_dir_flipped));
    }

    *min_is_flipped = flipped < direct;
    if(*min_is_flipped)
    {
        return flipped / (float)(nb_directions);
    }

    return direct / (float)(nb_directions);
}

float mdf_avg(const float4* s, const float4* t,
              const float s_scaling, const float t_scaling,
              const int nb_points, bool* min_is_flipped)
{
    float direct = 0.0f;
    float flipped = 0.0f;
    for(int i = 0; i < nb_points; ++i)
    {
        direct += fast_distance(s_scaling*s[i].xyz, t_scaling*t[i].xyz);
        flipped += fast_distance(s_scaling*s[i].xyz, t_scaling*t[nb_points - 1 -i].xyz);
    }

    *min_is_flipped = flipped < direct;
    if(*min_is_flipped)
    {
        return flipped / (float)nb_points;
    }
    return direct / (float)nb_points;
}

int qb(const float4* resampled_track, const int nb_resample,
       const float max_deviation, const int nb_clusters,
       float4* cluster_track_sums, int* cluster_track_counts)
{
    float min_dist = FLT_MAX;
    int best_cluster_id = 0;
    bool needs_flip = false;
    float dist = 0.0f;
    for(int i = 0; i < nb_clusters; ++i)
    {
        // compute distance to each cluster
        bool _needs_flip;
        if(DIST_METRIC == MAD_MIN)
        {
            dist = mad_min(resampled_track,
                           &cluster_track_sums[i*nb_resample],
                           nb_resample, &_needs_flip);
        }
        else // if(DIST_METRIC == MDF_AVG)
        {
            dist = mdf_avg(resampled_track,
                           &cluster_track_sums[i*nb_resample],
                           1.0f, 1.0f/cluster_track_counts[i],
                           nb_resample, &_needs_flip);
        }

        if(dist < min_dist)
        {
            min_dist = dist;
            best_cluster_id = i;
            needs_flip = _needs_flip;
        }
    }

    if(min_dist > max_deviation && nb_clusters < MAX_N_CLUSTERS)
    {
        best_cluster_id = nb_clusters; // create new cluster
        cluster_track_counts[best_cluster_id] = 1;

        // initialize track sum to 0
        for(int i = 0; i < nb_resample; ++i)
        {
            cluster_track_sums[best_cluster_id*nb_resample+i] =
                (float4)(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }
    else // we increment the number of tracks in the best cluster
    {
        ++cluster_track_counts[best_cluster_id];
    }

    if(needs_flip)
    {
        if(DIST_METRIC == MAD_MIN)
        {
            for(int i = 0; i < nb_resample; ++i)
            {
                cluster_track_sums[(best_cluster_id + 1)*nb_resample - 1 - i] += -resampled_track[i];
            }
        }
        else // if(DIST_METRIC == MDF_AVG)
        {
            for(int i = 0; i < nb_resample; ++i)
            {
                cluster_track_sums[(best_cluster_id + 1)*nb_resample - 1 - i] += resampled_track[i];
            }
        }
    }
    else
    {
        for(int i = 0; i < nb_resample; ++i)
        {
            cluster_track_sums[best_cluster_id*nb_resample+i] += resampled_track[i];
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

        // resample track before clustering
        float4 resampled_track[N_RESAMPLE];
        if(DIST_METRIC == MAD_MIN)
        {
            resample_track_to_vecs(&all_points[strl_pts_offset],
                                   strl_nb_points, N_RESAMPLE,
                                   resampled_track);
        }
        else // if(DIST_METRIC == MDF_AVG)
        {
            resample_track_to_points(&all_points[strl_pts_offset],
                                     strl_nb_points, N_RESAMPLE,
                                     resampled_track);
        }

        strl_cluster_ids[i] = qb(resampled_track, N_RESAMPLE, MAX_DEVIATION, nb_clusters,
                                 cluster_track_sums, cluster_track_counts);

        if(strl_cluster_ids[i] == nb_clusters)
        {
            ++nb_clusters;
        }
    }

    // Merge clusters
    // first, copy centroids
    float4 centroids[MAX_N_CLUSTERS*N_RESAMPLE];
    int merged_cluster_ids[MAX_N_CLUSTERS];
    float4 merged_clusters_track_sums[MAX_N_CLUSTERS*N_RESAMPLE];
    int merged_clusters_track_counts[MAX_N_CLUSTERS];
    uint nb_merged_clusters = 0;
    if(MERGE_CLUSTERS > 0)
    {
        for(uint i = 0; i < nb_clusters; ++i)
        {
            for(uint point_id = 0; point_id < N_RESAMPLE; ++point_id)
            {
                // centroids are rescaled during copy
                centroids[i*N_RESAMPLE+point_id] = cluster_track_sums[i*N_RESAMPLE+point_id]
                                                 / cluster_track_counts[i];
            }
        }

        for(uint i = 0; i < nb_clusters; ++i)
        {
            merged_cluster_ids[i] = qb(&centroids[i*N_RESAMPLE],
                                    N_RESAMPLE, MERGE_DIST_RATIO*MAX_DEVIATION,
                                    nb_merged_clusters,
                                    merged_clusters_track_sums,
                                    merged_clusters_track_counts);

            if(merged_cluster_ids[i] == nb_merged_clusters)
            {
                ++nb_merged_clusters;
            }
        }
    }
    else
    {
        for(uint i = 0; i < nb_clusters; ++i)
        {
            merged_cluster_ids[i] = i;
        }
        nb_merged_clusters = nb_clusters;
    }

    // on copie les streamlines cluster ids
    for(uint i = 0; i < nb_streamlines; ++i)
    {
        out_cluster_ids[first_strl_offset+i] = merged_cluster_ids[strl_cluster_ids[i]];
    }
}