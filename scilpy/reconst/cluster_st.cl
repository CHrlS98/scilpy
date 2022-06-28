/*
Cluster short tracks tractogram by running Quickbundles inside each voxel.

TODO: All streamlines do not need to be pushed on the GPU at the same time.
*/

// TODO: Change these values
#define N_RESAMPLE 6
#define MAX_N_CLUSTERS 10
#define MAX_NB_STRL 0
#define MAX_DEVIATION 1.0f

uint get_nb_points(uint index, __global const uint* strl_offsets)
{
    return strl_offsets[index + 1] - strl_offsets[index];
}

uint get_nb_streamlines(uint voxel_id,
                        __global const uint* strl_per_vox_offsets)
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
            track, &cluster_track_sums[i*N_RESAMPLE],
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


__kernel void cluster_per_voxel(__global const float4* all_points,
                                __global const uint* strl_pts_offsets,
                                __global const uint* strl_per_vox_offsets,
                                __global uint* out_nb_clusters)
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
    int nb_clusters = 0;


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
    out_nb_clusters[voxel_id] = nb_clusters;
}