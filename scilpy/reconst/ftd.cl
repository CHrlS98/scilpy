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
#define N_SEEDS_PER_VOX 0
#define MAX_LENGTH 0
#define FORWARD_ONLY false

// CONSTANTS
#define FLOAT_TO_BOOL_EPSILON 0.1f
#define NULL_SF_EPS 0.0001f

// pseudo random number generator
void rand_xorshift(uint* rng_state)
{
    rng_state[0] ^= (rng_state[0] << 13);
    rng_state[0] ^= (rng_state[0] >> 17);
    rng_state[0] ^= (rng_state[0] << 5);
}

int get_flat_index(const int x, const int y, const int z, const int w,
                   const int xLen, const int yLen, const int zLen)
{
    return x + y * xLen + z * xLen * yLen + w * xLen * yLen * zLen;
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

void sh_to_sf(const float* sh_coeffs, __global const float* sh_to_sf_mat,
              const bool is_first_step, __global const float3* vertices,
              const float3 last_dir, const float max_cos_theta, float* sf_coeffs)
{
    for(int u = 0; u < N_DIRS; ++u)
    {
        const float3 vertice = vertices[u];

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

int propagate(float3 last_pos, float3 last_dir,
              int current_length, uint* rng_state,
              __global const float* tracking_mask,
              __global const float* sh_coeffs,
              __global const float3* vertices,
              __global const float* sh_to_sf_mat,
              float3* out_track)
{
    bool is_valid = is_valid_pos(tracking_mask, last_pos);
    while(current_length < MAX_LENGTH && is_valid)
    {
        // save current streamline position
        out_track[current_length] = last_pos;

        // 2. Sample SF at position.
        float odf_sh[IM_N_COEFFS];
        float odf_sf[N_DIRS];
        get_value_nn(sh_coeffs, IM_N_COEFFS, last_pos, odf_sh);
        sh_to_sf(odf_sh, sh_to_sf_mat, current_length == 0, vertices,
                 last_dir, MIN_COS_THETA, odf_sf);

        rand_xorshift(rng_state);
        const float randv = (float)rng_state[0] / (float)UINT_MAX;
        const int vert_indice = sample_sf(odf_sf, randv);
        if(vert_indice > 0)
        {
            const float3 direction = vertices[vert_indice];

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
        ++current_length;
    }
    return current_length;
}

// TODO: Test tracking before QB implementation.
__kernel void track(__global const uint3* voxel_ids,
                    __global const float* sh_coeffs,
                    __global const float* mask,
                    __global const float* sh_to_sf_mat,
                    __global const float3* vertices,
                    __global float* out_ftd,
                    __global uint* out_nb_bundles)
{
    // 1. Get seed position from global_id.
    const size_t global_id = get_global_id(0);
    const uint3 voxel_id = voxel_ids[global_id];

    uint rng_state = global_id + 1;

    // tableau de nseeds_per_vox x max_length float3
    float3 tracks[N_SEEDS_PER_VOX][MAX_LENGTH];
    for(int track_id = 0; track_id < N_SEEDS_PER_VOX; ++track_id)
    {
        // generate seeding position
        rand_xorshift(&rng_state);
        const float xrand = (float)rng_state / (float)UINT_MAX;
        rand_xorshift(&rng_state);
        const float yrand = (float)rng_state / (float)UINT_MAX;
        rand_xorshift(&rng_state);
        const float zrand = (float)rng_state / (float)UINT_MAX;
        const float3 seed_pos = {
            (float)voxel_id.x + xrand,
            (float)voxel_id.y + yrand,
            (float)voxel_id.z + zrand
        };

        // track from seed position
        float3 last_pos = seed_pos;
        float3 last_dir;
        int current_length = 0;

        // forward track
        // TODO: propagation maximum de 1/2*MAX_LENGTH par direction!
        current_length = propagate(last_pos, last_dir, current_length,
                                   &rng_state, mask, sh_coeffs, vertices,
                                   sh_to_sf_mat,  &tracks[track_id][0]);

        // reverse streamline for backward tracking
        if(current_length > 1 && current_length < MAX_LENGTH && !FORWARD_ONLY)
        {
            // reset last direction to initial direction
            last_dir = tracks[track_id][0] - tracks[track_id][1];
            last_dir = normalize(last_dir);

            // reverse streamline so the output is continuous
            reverse_streamline(current_length, &tracks[track_id][0]);

            // track backward
            current_length = propagate(last_pos, last_dir, current_length,
                                   &rng_state, mask, sh_coeffs, vertices,
                                   sh_to_sf_mat,  &tracks[track_id][0]);
        }
        // TODO: Quickbundle ici.
    }
    // TODO: Un coup qu'on a nos bundles, on calcule une FTD par bundle.
}
