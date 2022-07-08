/*
Cluster short tracks tractogram by running Quickbundles inside each voxel.
*/

#define N_RESAMPLE 6
#define MAX_N_CLUSTERS 10
#define MAX_NB_STRL 0
#define MAX_DEVIATION 0.7f
#define ABS_THRESHOLD 10
#define REL_THRESHOLD 0.1f

uint get_nb_points(uint index, __global const uint* strl_offsets)
{
    return strl_offsets[index + 1] - strl_offsets[index];
}

uint get_nb_streamlines(uint voxel_id, __global const uint* strl_per_vox_offsets)
{
    return strl_per_vox_offsets[voxel_id + 1] - strl_per_vox_offsets[voxel_id];
}

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

void resample_track(__global const float4* in_track, const int nb_points, float4* out_track)
{
    const float delta_t = 1.0f / (float)(N_RESAMPLE - 1);
    for (int i = 0; i < N_RESAMPLE; i++)
    {
        out_track[i] = interp_along_track(in_track, nb_points, i*delta_t);
    }
}

/*
The minimum mean angular deviation is the average deviation (in radians)
between the direction at each step along two streamlines.
*/
float min_mean_angular_deviation(const float4* s, const float4* t,
                                 bool* min_is_flipped)
{
    float direct = 0.0f;
    float flipped = 0.0f;
    float3 s_dir;
    float3 t_dir;
    float3 t_dir_flipped;
    for(int i = 0; i < N_RESAMPLE - 1; ++i)
    {
        s_dir = normalize(s[i+1].xyz - s[i].xyz);
        t_dir = normalize(t[i+1].xyz - t[i].xyz);
        t_dir_flipped = normalize(t[N_RESAMPLE - 2 - i].xyz -
                                  t[N_RESAMPLE - 1 - i].xyz);
        direct += acos(dot(s_dir, t_dir));
        flipped += acos(dot(s_dir, t_dir_flipped));
    }

    *min_is_flipped = flipped < direct;
    if(*min_is_flipped)
    {
        return flipped / (float)(N_RESAMPLE - 1);
    }

    return direct / (float)(N_RESAMPLE - 1);
}

// return the id of the closest cluster
int quick_bundle(__global const float4* track,
                 const int nb_points,
                 const float max_deviation,
                 const int nb_clusters,
                 float4* cluster_track_sums,
                 int* cluster_track_counts)
{
    // 1. resample track
    float4 resampled_track[N_RESAMPLE];
    resample_track(track, nb_points, resampled_track);

    // 2. compare the resampled track to all cluster centroids
    float min_dist = FLT_MAX;
    int best_cluster_id = 0;
    bool needs_flip = false;
    for(int i = 0; i < nb_clusters; ++i)
    { // we won't reach the inside of the loop if there are no clusters

        // compute distance to each cluster
        bool _needs_flip;
        const float dist = min_mean_angular_deviation(
            resampled_track, &cluster_track_sums[i*N_RESAMPLE],
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
        // printf("%s, %f\n", "Create new bundle", min_mdf);

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

    // 4. add the resampled track to the cluster track sum
    for (int i = 0; i < N_RESAMPLE; i++)
    {
        if(needs_flip)
        {
            cluster_track_sums[(best_cluster_id + 1)*N_RESAMPLE - 1 - i] += resampled_track[i];
        }
        else
        {
            cluster_track_sums[best_cluster_id*N_RESAMPLE+i] += resampled_track[i];
        }
    }
    return best_cluster_id;
}


int merge_clusters(int nb_clusters, float4* cluster_track_sums,
                   int* cluster_track_counts)
{
    // tableau de merged ids initialisé à l'id de chaque cluster
    /*
    int centroids_cluster_ids[MAX_N_CLUSTERS];
    for(int i = 0; i < MAX_N_CLUSTERS; ++i)
    {
        centroids_cluster_ids[i] = i;
    }
    int nb_merged_clusters = 0;
    */

    bool merged_clusters = true;
    bool flip_required = false;
    while(merged_clusters)
    {
        merged_clusters = false;
        for(uint ref_cluster_id = 0;
            ref_cluster_id < nb_clusters - 2 && !merged_clusters;
            ++ref_cluster_id)
        {
            const float4* s = &cluster_track_sums[ref_cluster_id*N_RESAMPLE];
            for(uint curr_cluster_id = ref_cluster_id + 1;
                curr_cluster_id < nb_clusters && !merged_clusters;
                ++curr_cluster_id)
            {
                const float4* t = &cluster_track_sums[curr_cluster_id*N_RESAMPLE];
                const float dist = min_mean_angular_deviation(s, t, &flip_required);
                if(dist < MAX_DEVIATION)
                {
                    const int new_track_count = cluster_track_counts[ref_cluster_id]
                                              + cluster_track_counts[curr_cluster_id];

                    // FIXME: remove weights
                    const float s_weight = 1.0f;
                    const float t_weight = 1.0f;
                    for(uint point_id = 0; point_id < N_RESAMPLE; ++point_id)
                    {
                        const uint ref_cluster_point_id = ref_cluster_id*N_RESAMPLE+point_id;
                        const uint curr_cluster_point_id =
                            flip_required ?
                            (curr_cluster_id+1)*N_RESAMPLE - point_id - 1 :
                            curr_cluster_id*N_RESAMPLE + point_id;

                        // update centroid
                        cluster_track_sums[ref_cluster_id*N_RESAMPLE+point_id] =
                            s_weight * cluster_track_sums[ref_cluster_id*N_RESAMPLE+point_id] +
                            t_weight * cluster_track_sums[curr_cluster_point_id];
                    }
                    // update track count
                    cluster_track_counts[ref_cluster_id] = new_track_count;

                    // decaler tout ce qui vient après curr_cluster_id
                    for(uint shift_id = curr_cluster_id + 1; shift_id < nb_clusters; ++shift_id)
                    {
                        cluster_track_counts[shift_id - 1] = cluster_track_counts[shift_id];
                        for(uint point_id = 0; point_id < N_RESAMPLE; ++point_id)
                        {
                            cluster_track_sums[(shift_id-1)*N_RESAMPLE+point_id] =
                                cluster_track_sums[shift_id*N_RESAMPLE+point_id];
                        }
                    }

                    // TODO: assign strl to new cluster

                    // decrease the total number of clusters
                    --nb_clusters;
                    merged_clusters = true;
                }
            }
        }
    }
    return nb_clusters;
}


__kernel void cluster_per_voxel(__global const float4* all_points,
                                __global const uint* strl_pts_offsets,
                                __global const uint* strl_per_vox_offsets,
                                __global uint* out_nb_clusters,
                                __global float4* out_centroids_points)
{
    const int voxel_id = get_global_id(0);
    const uint first_strl_offset = strl_per_vox_offsets[voxel_id];
    const uint nb_streamlines = get_nb_streamlines(voxel_id, strl_per_vox_offsets);

    // QB clustering data structures
    // Cluster ID of each streamline to cluster
    int strl_cluster_ids[MAX_NB_STRL];

    // Non-normalized centroid of each cluster
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

        strl_cluster_ids[i] = quick_bundle(&all_points[strl_pts_offset], strl_nb_points,
                                           MAX_DEVIATION, nb_clusters, cluster_track_sums,
                                           cluster_track_counts);
        if(strl_cluster_ids[i] == nb_clusters)
        {
            ++nb_clusters;
        }
    }

    // Merge clusters
    nb_clusters = merge_clusters(nb_clusters, cluster_track_sums, cluster_track_counts);

    // Absolute threshold: A valid cluster is expected to have
    // at least 10 short tracks

    // Relative threshold: A cluster is valid if it contains at least
    // rel_th*max(cluster_tracks_counts) short tracks
    int max_n_tracks = 0;
    for(int i = 0; i < nb_clusters; ++i)
    {
        if(cluster_track_counts[i] > max_n_tracks)
        {
            max_n_tracks = cluster_track_counts[i];
        }
    }
    const int threshold = max(ABS_THRESHOLD, (int)(REL_THRESHOLD * (float)max_n_tracks));

    // Copy centroids to global memory
    // because clusters can be discarded based on absolute and relative thresholds
    // we use a contiguous_cluster_id to save streamlines.
    uint contiguous_cluster_id = 0;
    for(uint cluster_id = 0; cluster_id < nb_clusters; ++cluster_id)
    {
        // apply threshold based on track count
        if(cluster_track_counts[cluster_id] >= threshold)
        {
            for(uint point_id = 0; point_id < N_RESAMPLE; ++point_id)
            {
                const uint track_sums_offset = cluster_id * N_RESAMPLE + point_id;
                const uint out_centroids_offset = voxel_id * MAX_N_CLUSTERS * N_RESAMPLE
                                                + contiguous_cluster_id * N_RESAMPLE
                                                + point_id;
                out_centroids_points[out_centroids_offset] = cluster_track_sums[track_sums_offset]
                                                        / (float)cluster_track_counts[cluster_id];
            }
            ++contiguous_cluster_id;
        }
    }
    out_nb_clusters[voxel_id] = contiguous_cluster_id;
}