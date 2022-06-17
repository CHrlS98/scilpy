/*
OpenCL kernel for estimating fiber trajectory distributions
numerically.
*/

// Compiler definitions with placeholder values
#define IM_X_DIM 0
#define IM_Y_DIM 0
#define IM_Z_DIM 0
#define IM_N_COEFFS 0
#define N_DIRS 1

#define MIN_COS_THETA 0
#define STEP_SIZE 0
#define N_VOX 0
#define N_SEEDS_PER_VOX 0
#define MAX_N_CLUSTERS 5
#define QB_MDF_THRESHOLD 0.0f
#define QB_MDF_MERGE_THRESHOLD 0.0f
#define QB_N_TRACKS_THRESHOLD 0
#define MIN_LENGTH 0
#define MAX_LENGTH 0
#define FORWARD_ONLY false

// CONSTANTS
#define FLOAT_TO_BOOL_EPSILON 0.1f
#define NULL_SF_EPS 0.0001f
#define N_RESAMPLE 10

// pseudo random number generator
void rand_xorshift(uint* rng_state)
{
    rng_state[0] ^= (rng_state[0] << 13);
    rng_state[0] ^= (rng_state[0] >> 17);
    rng_state[0] ^= (rng_state[0] << 5);
}

float3 interp_along_track(const float3* track, const int length, float t)
{
    const float t_length = (float)length * t;
    const int t_int_floor = (int)t_length;
    if(t_int_floor == 0)
        return track[0];

    const int t_int_ceil = t_int_floor + 1;
    if(t_int_ceil > length - 1)
        return track[length - 1];
    
    const float3 p_prev = track[t_int_floor];
    const float3 p_next = track[t_int_ceil];

    const float r = t_length - floor(t_length);
    return r*p_next + (1.0f - r)*p_prev;
}

void resample_track(const float3* in_track,
                    const int length,
                    float3* out_track)
{
    const float delta_t = 1.0f / (float)(N_RESAMPLE - 1);
    // printf("%s\n", "resampled track");
    for (int i = 0; i < N_RESAMPLE; i++)
    {
        out_track[i] = interp_along_track(in_track, length, i*delta_t);
        // printf("f3 = %2.2v4hlf\n", out_track[i]);
    }
}

float compute_mdf(const float3* s, const float s_scale,
                  const float3* t, const float t_scale,
                  bool* hmmm_flip)
{
    float direct = 0.0f;
    float flipped = 0.0f;

    for(int i = 0; i < N_RESAMPLE; i++)
    {
        direct += fast_distance(s[i]*s_scale, t[i]*t_scale);
        flipped += fast_distance(s[i]*s_scale, t[-i + N_RESAMPLE - 1]*t_scale);
    }

    if(flipped < direct)
    {
        *hmmm_flip = true;
        return 1.0f / (float)N_RESAMPLE * flipped;
    }
    *hmmm_flip = false;
    return 1.0f / (float)N_RESAMPLE * direct;
}

void reverse_streamline(const int num_strl_points,
                        float3* track)
{
    for(int i = 0; i < (int)(num_strl_points/2); ++i)
    {
        const float3 temp_pt = track[i];
        track[i] = track[num_strl_points - 1 - i];
        track[num_strl_points - 1 - i] = temp_pt;
    }
}

// return the id of the closest cluster
int quick_bundle(float3* track,  // we could flip it if necessary
                 const int length,
                 const int n_clusters,
                 float3* cluster_track_sums,
                 int* cluster_track_counts)
{
    // 1. resample track
    float3 resampled_track[N_RESAMPLE];
    resample_track(track, length, resampled_track);

    // 2. compare the resampled track to all cluster centroids
    float min_mdf = FLT_MAX;
    int min_cluster_id = 0;
    bool min_needs_flip = false;
    for(int i = 0; i < n_clusters; ++i)
    {
        const float t_scale = 1.0f / (float)cluster_track_counts[i];

        // compute mdf and optionally flip the resampled track
        bool needs_flip;
        const float mdf = compute_mdf(&cluster_track_sums[i*N_RESAMPLE], t_scale,
                                      resampled_track, 1.0f, &needs_flip);
        // if(i == 0) printf("%f\n", mdf);

        if(mdf < min_mdf)
        {
            min_mdf = mdf;
            min_cluster_id = i;
            min_needs_flip = needs_flip;
        }
    }

    // when required, reverse streamlines
    if(min_needs_flip)
    {
        reverse_streamline(N_RESAMPLE, resampled_track);
        reverse_streamline(length, track);
    }

    // 3. if min_mdf > QB_THRESHOLD (or there are no
    // clusters) then we need to add a new cluster
    if(min_mdf > QB_MDF_THRESHOLD && n_clusters < MAX_N_CLUSTERS)
    {
        // printf("%s, %f\n", "Create new bundle", min_mdf);

        min_cluster_id = n_clusters; // create new cluster
        cluster_track_counts[min_cluster_id] = 1;
    }
    else
    {
        ++cluster_track_counts[min_cluster_id];
    }

    // 4. increment the cluster track count and add the
    // resampled track to the cluster track sum
    for (int i = 0; i < N_RESAMPLE; i++)
    {
        cluster_track_sums[min_cluster_id*N_RESAMPLE+i] += resampled_track[i];
    }
    return min_cluster_id;
}

int get_flat_index(const int x, const int y, const int z, const int w,
                   const int xLen, const int yLen, const int zLen)
{
    return x + y * xLen + z * xLen * yLen + w * xLen * yLen * zLen;
}

void copy_track_to_output(float3* track, uint global_id, uint track_id,
                          uint length, int cluster_id, __global float* out_tracks,
                          __global uint* out_nb_points, __global int* out_cluster_ids)
{
    for(uint i = 0; i < length; ++i)
    {
        out_tracks[get_flat_index(global_id*N_SEEDS_PER_VOX+track_id, i, 0,
                                  0, N_VOX*N_SEEDS_PER_VOX, MAX_LENGTH, 3)] = track[i].x;
        out_tracks[get_flat_index(global_id*N_SEEDS_PER_VOX+track_id, i, 1,
                                  0, N_VOX*N_SEEDS_PER_VOX, MAX_LENGTH, 3)] = track[i].y;
        out_tracks[get_flat_index(global_id*N_SEEDS_PER_VOX+track_id, i, 2,
                                  0, N_VOX*N_SEEDS_PER_VOX, MAX_LENGTH, 3)] = track[i].z;
    }
    out_nb_points[get_flat_index(global_id*N_SEEDS_PER_VOX+track_id, 0,
                                 0, 0, N_VOX*N_SEEDS_PER_VOX, 1, 1)] = length;
    out_cluster_ids[get_flat_index(global_id*N_SEEDS_PER_VOX+track_id, 0,
                                   0, 0, N_VOX*N_SEEDS_PER_VOX, 1, 1)] = cluster_id;
}

void sh_to_sf(const float* sh_coeffs, __global const float* sh_to_sf_mat,
              const bool is_first_step, __global const float4* vertices,
              const float3 last_dir, const float max_cos_theta, float* sf_coeffs)
{
    for(int u = 0; u < N_DIRS; ++u)
    {
        const float3 vertice = vertices[u].xyz;

        // all directions are valid for first step.
        bool is_valid = is_first_step;
        // if not in first step, we need to check that
        // we are inside the tracking cone
        if(!is_valid)
        {
            is_valid = dot(last_dir, vertice) > max_cos_theta;
        }

        sf_coeffs[u] = 0.0f;
        if(is_valid)
        {
            for(int j = 0; j < IM_N_COEFFS; ++j)
            {
                const float ylmu_inv = sh_to_sf_mat[get_flat_index(j, u, 0, 0,
                                                                   IM_N_COEFFS,
                                                                   N_DIRS, 1)];
                sf_coeffs[u] += ylmu_inv * sh_coeffs[j];
            }
            // clip negative values
            if(sf_coeffs[u] < 0.0f)
            {
                sf_coeffs[u] = 0.0f;
            }
        }
    }
}

void get_value_nn(__global const float* image, const int n_channels,
                  const float3 pos, float* value)
{
    const int x = (int)pos.x;
    const int y = (int)pos.y;
    const int z = (int)pos.z;
    for(int w = 0; w < n_channels; ++w)
    {
        value[w] = image[get_flat_index(x, y, z, w, IM_X_DIM, IM_Y_DIM, IM_Z_DIM)];
    }
}

bool is_valid_pos(__global const float* tracking_mask, const float3 pos)
{
    const bool is_inside_volume = pos.x >= 0.0 && pos.x < IM_X_DIM &&
                                  pos.y >= 0.0 && pos.y < IM_Y_DIM &&
                                  pos.z >= 0.0 && pos.z < IM_Z_DIM;

    if(is_inside_volume)
    {
        float mask_value[1];
        get_value_nn(tracking_mask, 1, pos, mask_value);
        const bool is_inside_mask = mask_value[0] > FLOAT_TO_BOOL_EPSILON;
        return is_inside_mask;
    }
    return false;
}

int sample_sf(const float* odf_sf, const float randv)
{
    float cumsum[N_DIRS];
    cumsum[0] = odf_sf[0];
    for(int i = 1; i < N_DIRS; ++i)
    {
        cumsum[i] = odf_sf[i] + cumsum[i - 1];
    }

    if(cumsum[N_DIRS - 1] < NULL_SF_EPS)
    {
        return -1;
    }

    const float where = (randv * cumsum[N_DIRS - 1]);
    int index = 0;
    while(cumsum[index] < where && index < N_DIRS)
    {
        ++index;
    }
    return index;
}

int propagate(float3 last_pos, float3 last_dir, int current_length,
              uint* rng_state, bool is_forward,
              __global const float* tracking_mask,
              __global const float* sh_coeffs,
              __global const float4* vertices,
              __global const float* sh_to_sf_mat,
              float3* out_track)
{
    bool is_valid = is_valid_pos(tracking_mask, last_pos);
    const int max_length = is_forward && !FORWARD_ONLY ? MAX_LENGTH / 2 : MAX_LENGTH;

    while(current_length < max_length && is_valid)
    {
        // 2. Sample SF at position.
        float odf_sh[IM_N_COEFFS];
        float odf_sf[N_DIRS];
        get_value_nn(sh_coeffs, IM_N_COEFFS, last_pos, odf_sh);
        sh_to_sf(odf_sh, sh_to_sf_mat, current_length == 1, vertices,
                 last_dir, MIN_COS_THETA, odf_sf);

        rand_xorshift(rng_state);
        const float randv = (float)rng_state[0] / (float)UINT_MAX;
        const int vert_indice = sample_sf(odf_sf, randv);
        if(vert_indice > 0)
        {
            const float3 direction = vertices[vert_indice].xyz;

            // 3. Try step.
            const float3 next_pos = last_pos + STEP_SIZE * direction;
            is_valid = is_valid_pos(tracking_mask, next_pos);
            last_dir = normalize(next_pos - last_pos);
            last_pos = next_pos;
        }
        else
        {
            is_valid = false;
        }

        if(is_valid)
        {
            // save current streamline position
            out_track[current_length] = last_pos;
            ++current_length;
        }
    }
    return current_length;
}

int track(float3 voxel_id, uint* rng_state,
          const __global float* mask,
          const __global float* sh_coeffs,
          const __global float4* vertices,
          const __global float* sh_to_sf_mat,
          float3* out_track)
{
    // generate seeding position
    rand_xorshift(rng_state);
    const float xrand = (float)rng_state[0] / (float)UINT_MAX;
    rand_xorshift(rng_state);
    const float yrand = (float)rng_state[0] / (float)UINT_MAX;
    rand_xorshift(rng_state);
    const float zrand = (float)rng_state[0] / (float)UINT_MAX;
    const float3 seed_pos = voxel_id + (float3)(xrand, yrand, zrand);

    // track from seed position
    float3 last_pos = seed_pos;
    float3 last_dir;

    // initialize streamline with seed position
    out_track[0] = last_pos;
    int current_length = 1;

    // forward track
    current_length = propagate(last_pos, last_dir, current_length,
                               rng_state, true, mask, sh_coeffs, vertices,
                               sh_to_sf_mat,  out_track);

    // reverse streamline for backward tracking
    if(current_length > 1 && current_length < MAX_LENGTH && !FORWARD_ONLY)
    {
        // reset last direction to initial direction
        last_dir = out_track[0] - out_track[1];
        last_dir = normalize(last_dir);

        // reverse streamline so the output is continuous
        reverse_streamline(current_length, &out_track[0]);

        // track backward
        current_length = propagate(last_pos, last_dir, current_length,
                                   rng_state, false, mask, sh_coeffs,
                                   vertices, sh_to_sf_mat, out_track);
    }
    return current_length;
}

__kernel void main(__global const float4* voxel_ids,
                   __global const float* sh_coeffs,
                   __global const float* mask,
                   __global const float* sh_to_sf_mat,
                   __global const float4* vertices,
                   __global float* out_tracks,
                   __global uint* out_nb_points,
                   __global int* out_cluster_ids)
{
    // 1. Get seed position from global_id.
    const size_t global_id = get_global_id(0);
    const float4 voxel_id = voxel_ids[global_id];

    /*
    if(global_id != 5)
    {
        return;
    }
    */

    // initialize random number generator
    // NOTE: Can't be 0 because xorshift won't work with 0.
    uint rng_state = global_id + 1;

    // Tracking variables.
    float3 tracks[N_SEEDS_PER_VOX][MAX_LENGTH];
    int n_points[N_SEEDS_PER_VOX];

    // Quickbundle clustering variables.
    int cluster_ids[N_SEEDS_PER_VOX];
    int cluster_track_counts[MAX_N_CLUSTERS];
    float3 cluster_track_sums[MAX_N_CLUSTERS*N_RESAMPLE];
    int n_clusters = 0;

    for(int track_id = 0; track_id < N_SEEDS_PER_VOX; ++track_id)
    {
        // 1. Track streamline.
        n_points[track_id] = track(voxel_id.xyz, &rng_state,
                                   mask, sh_coeffs, vertices,
                                   sh_to_sf_mat, tracks[track_id]);

        // 2. Quickbundle clustering.
        if(n_points[track_id] > MIN_LENGTH)
        {
            cluster_ids[track_id] = quick_bundle(tracks[track_id], n_points[track_id],
                                                 n_clusters, cluster_track_sums,
                                                 cluster_track_counts);
            if(cluster_ids[track_id] > n_clusters -1)
            {
                ++n_clusters;
            }
        }
        else
        {
            cluster_ids[track_id] = -1; // invalid track flag
        }
    }
    // TODO: Un coup qu'on a TOUS nos bundles, on calcule une FTD par bundle.
    // MAYBE: Faire sur le CPU pour utiliser numpy for matrix inversion.

    // Merger les bundles similaires
    // !!! FAUDRAIT LE FAIRE AU FUR ET A MESURE POUR QUE LES
    // PROCHAINES COMPARAISONS UTILISENT LES NOUVEAUX CENTROIDES !!!
    bool cluster_needs_merge[N_RESAMPLE][N_RESAMPLE];
    bool cluster_needs_flip[N_RESAMPLE][N_RESAMPLE];
    for(int ref_cluster_id = 0; ref_cluster_id < n_clusters - 2; ++ref_cluster_id)
    {
        for(int cluster_id = ref_cluster_id + 1; cluster_id < n_clusters; ++cluster_id)
        {
            const float3* ref_centroid = &cluster_track_sums[ref_cluster_id*N_RESAMPLE];
            const float ref_scale = 1.0f / (float)cluster_track_counts[ref_cluster_id];
            const float3* other_centroid = &cluster_track_sums[cluster_id*N_RESAMPLE];
            const float other_scale = 1.0f / (float)cluster_track_counts[cluster_id];

            bool needs_flip = false;
            const float mdf = compute_mdf(ref_centroid, ref_scale,
                                          other_centroid, other_scale,
                                          &needs_flip);
        }
    }

    // Cleaner les bundles sous-representes.
    int n_invalid_clusters = 0;
    int invalid_cluster_ids[MAX_N_CLUSTERS];
    for(int cluster_id = 0; cluster_id < MAX_N_CLUSTERS; ++cluster_id)
    {
        if(cluster_track_counts[cluster_id] < QB_N_TRACKS_THRESHOLD)
        {
            invalid_cluster_ids[n_invalid_clusters++] = cluster_id;
        }
    }

    for(int track_id = 0; track_id < N_SEEDS_PER_VOX; ++track_id)
    {
        for(int cluster_id = 0; cluster_id < n_invalid_clusters; ++cluster_id)
        {
            const int invalid_cluster_id = invalid_cluster_ids[cluster_id];
            if(cluster_ids[track_id] == invalid_cluster_id)
            {
                cluster_ids[track_id] = -1; // invalid flag
            }
        }
    }

    // copy tracks for debug
    for(int track_id = 0; track_id < N_SEEDS_PER_VOX; ++track_id)
    {
        copy_track_to_output(tracks[track_id], global_id, track_id,
                             n_points[track_id], cluster_ids[track_id],
                             out_tracks, out_nb_points,
                             out_cluster_ids);
    }
}
