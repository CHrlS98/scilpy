/*
OpenCL kernel code for tracking over short-tracks tractograms.

NOTES:
* streamlines have maximum length;
* streamlines are forward- and backward-tracked;
*/

// placeholder values for compiler definitions
#define NUM_CELLS 0
#define SEARCH_RADIUS 0.0f
#define MAX_DENSITY 0
#define MAX_SEARCH_NEIGHBOURS 0
#define CELLS_XMAX 0
#define CELLS_YMAX 0
#define CELLS_ZMAX 0
#define MAX_IN_ST_LEN 10
#define MAX_OUT_STRL_LEN 0
#define NUM_STEPS_PER_ITER 10
#define STEP_SIZE 0.0f
#define MIN_COS_ANGLE 0.0f

const int4 CELLS_GRID_DIMS = {CELLS_XMAX, CELLS_YMAX, CELLS_ZMAX, 0};


void rand_xorshift(uint* rng_state)
{
    rng_state[0] ^= (rng_state[0] << 13);
    rng_state[0] ^= (rng_state[0] >> 17);
    rng_state[0] ^= (rng_state[0] << 5);
}


float normal_dist(float x, float mu, float sigma)
{
    const float norm = 1.0f / (sigma * sqrt(2.0f*M_PI));
    return norm * exp(-0.5f * pow((x - mu)/sigma, 2.0f));
}


int resample_st(__global const float4* all_st_points, const int st_offset,
                const int st_num_pts, const float step_size,
                const int resampled_st_max_count, float4* resampled_st)
{
    // resampled_st contains at most NUM_STEPS_PER_ITER points
    // first point on resampled short-track is the first point
    // on the original short-track
    int current_index = st_offset;
    resampled_st[0] = all_st_points[current_index];
    int num_resampled_pts = 1;

    bool can_continue = true;
    bool is_last_segment = false;
    while(can_continue && num_resampled_pts < resampled_st_max_count)
    {
        // NEEDS TESTING: This condition might not work!
        if(current_index == st_offset + st_num_pts - 2)
        {
            // once toggled to true, won't go back to false.
            is_last_segment = true;
        }

        if(!is_last_segment)
        {
            while(distance(resampled_st[num_resampled_pts-1].xyz,
                           all_st_points[current_index+1].xyz) < step_size)
            {
                // gives indice of last point included inside step_size.
                ++current_index;
            }
        }
        else
        {
            can_continue = distance(all_st_points[st_offset + st_num_pts - 1].xyz,
                                    resampled_st[num_resampled_pts - 1].xyz) > step_size;
        }

        const float3 v = all_st_points[current_index+1].xyz
                       - all_st_points[current_index].xyz;
        const float3 x0 = all_st_points[current_index].xyz;
        const float3 p = resampled_st[num_resampled_pts - 1].xyz;

        // quadratic equation coefficients
        const float a = v.x*v.x + v.y*v.y + v.z*v.z;
        const float b = 2.0f * (v.x * x0.x + v.y * x0.y + v.z * x0.z -
                                p.x * v.x - p.y * v.y - p.z * v.z);
        const float c = x0.x*x0.x + x0.y*x0.y + x0.z*x0.z
                      - 2.0f * (p.x * x0.x + p.y * x0.y + p.z * x0.z)
                      + p.x*p.x + p.y*p.y + p.z*p.z - step_size*step_size;
        // solve to find next point of resampled short-track
        const float t = (-b + sqrt(b*b - 4.0f*a*c))/(2.0f*a);
        resampled_st[num_resampled_pts++].xyz = x0 + t*v;
    }

    return num_resampled_pts;
}


int get_length_from_offset(const size_t index,
                           __global const int* all_st_offsets)
{
    return all_st_offsets[index + 1] - all_st_offsets[index];
}


int dichotomic_search_cell(const int id, __global const int* cell_ids)
{
    size_t search_range[2] = {0, NUM_CELLS-1};
    // search until we have only two values left
    while(search_range[1] - search_range[0] > 1)
    {
        const size_t mid_id = (search_range[0] + search_range[1]) / 2;
        if(id == cell_ids[mid_id])
        {
            return mid_id;
        }
        else if(id < cell_ids[mid_id])
        {
            search_range[1] = mid_id;
        }
        else
        {
            search_range[0] = mid_id;
        }
    }

    // the value is one of the two bounds
    if(id == cell_ids[search_range[0]])
    {
        return search_range[0];
    }
    else if(id == cell_ids[search_range[1]])
    {
        return search_range[1];
    }

    // or the value is not present in the cells array
    return -1;
}


int map_to_cell_id(const float4 point)
{
    const float4 point_in_cells_space = point / SEARCH_RADIUS;

    if(point_in_cells_space.x >= 0.0f && point_in_cells_space.x < (float)CELLS_GRID_DIMS.x
    && point_in_cells_space.y >= 0.0f && point_in_cells_space.y < (float)CELLS_GRID_DIMS.y
    && point_in_cells_space.z >= 0.0f && point_in_cells_space.z < (float)CELLS_GRID_DIMS.z)
    {
        const int3 cell_id_3dimensional = {(int)point_in_cells_space.x,
                                           (int)point_in_cells_space.y,
                                           (int)point_in_cells_space.z};

        // ravel 3d index in 1d
        return cell_id_3dimensional.z * CELLS_GRID_DIMS.y * CELLS_GRID_DIMS.x +
            cell_id_3dimensional.y * CELLS_GRID_DIMS.x + cell_id_3dimensional.x;
    }
    // if we are outside the cells grid, we are without a doubt outside the trackable
    // voxels, just return -1
    return -1;
}


bool contains_unsorted(const int value, const int* array, const size_t len)
{
    for(size_t l = 0; l < len; ++l)
    {
        if(array[l] == value)
        {
            return true;
        }
    }
    return false;
}


int search_neighbours(const float4 pos, int* neighbours)
{
    int num_neighbours = 0;
    for(int x_offset = -1; x_offset <= 1; ++x_offset)
    {
        for(int y_offset = -1; y_offset <= 1; ++y_offset)
        {
            for(int z_offset = -1; z_offset <= 1; ++z_offset)
            {
                // new position is inside image range
                const float4 new_position = {
                    pos.x + x_offset*SEARCH_RADIUS,
                    pos.y + y_offset*SEARCH_RADIUS,
                    pos.z + z_offset*SEARCH_RADIUS,
                    1.0f};
                const int cell_id = map_to_cell_id(new_position);
                if(cell_id != -1) // outside image bounds
                {
                    // devrait pas donner de duplicats à moins d'erreurs d'arrondi.
                    neighbours[num_neighbours++] = cell_id;
                }
            }
        }
    }
    return num_neighbours;
}


int get_valid_trajectories(const float4 last_pos, const int num_neighbours, const int* neighbour_cells,
                            __global const int* cell_ids, __global const int* cell_st_counts,
                            __global const int* cell_st_offsets, __global const int* cell_st_ids,
                            __global const int* all_st_offsets, __global const float4* all_st_points,
                            int* unique_st)
{
    int num_valid_st = 0;

    // Go through all neighbour cells and add short-track ids
    for(int n_id = 0; n_id < num_neighbours; ++n_id)
    {
        const int neigh_cell_id = neighbour_cells[n_id];
        const int current_cell_index = dichotomic_search_cell(neigh_cell_id, cell_ids);
        if(current_cell_index > -1)  // cell contains at least one seed
        {
            const int st_count = cell_st_counts[current_cell_index];
            const int st_offset = cell_st_offsets[current_cell_index];

            // iterate through all streamlines in current cell
            for(int st_i = st_offset; st_i < st_offset + st_count; ++st_i)
            {
                const int st_id = cell_st_ids[st_i];
                const int st_offset = all_st_offsets[st_id];
                const float4 first_point = all_st_points[st_offset];
                if(distance(last_pos.xyz, first_point.xyz) < SEARCH_RADIUS)
                {
                    unique_st[num_valid_st++] = st_id;
                }
            }
        }
    }
    return num_valid_st;
}


int get_next_directions(const float4 curr_pos, const float4 prev_dir,
                       __global const int* cell_ids, __global const int* cell_st_counts,
                       __global const int* cell_st_offsets, __global const int* cell_st_ids,
                       __global const int* all_st_offsets, __global const float4* all_st_points,
                       uint* rng_state, float4* out_dirs)
{
    int neighbour_cells[MAX_SEARCH_NEIGHBOURS];
    const int num_neighbours = search_neighbours(curr_pos, neighbour_cells);

    int valid_st[MAX_SEARCH_NEIGHBOURS*MAX_DENSITY];
    const int num_valid_st = get_valid_trajectories(curr_pos,
        num_neighbours, neighbour_cells, cell_ids, cell_st_counts,
        cell_st_offsets, cell_st_ids, all_st_offsets, all_st_points,
        valid_st);

    // zero-fill trajectory_weights array
    int out_dirs_len = 0;
    for(int i_dir = 0; i_dir < NUM_STEPS_PER_ITER; ++i_dir)
    {
        out_dirs[i_dir] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    }

    // iterate through all valid streamlines
    // start from random offset
    rand_xorshift(rng_state);
    const float rand_val = (float)rng_state[0] / (float)UINT_MAX;
    const int rand_offset = (int)(rand_val * num_valid_st);

    int num_st_in_avg = 0;
    for(int i_valid_st  = 0; i_valid_st < num_valid_st; ++i_valid_st)
    {
        const int st_id = valid_st[(rand_offset + i_valid_st) % num_valid_st];
        const int st_offset = all_st_offsets[st_id];

        // then we need to test that it is under the maximum curvature threshold
        // using the first direction (second - first points)
        const float3 first_dir = normalize(all_st_points[st_offset + 1].xyz -
                                           all_st_points[st_offset].xyz);

        // deviation angle test
        if(dot(first_dir, prev_dir.xyz) > MIN_COS_ANGLE)
        {
            const int st_length = get_length_from_offset(st_id, all_st_offsets);

            // resample st with constant step_size
            float4 resampled_st[NUM_STEPS_PER_ITER + 1];

            int resampled_st_num_pts = out_dirs_len > 0 ? out_dirs_len + 1: NUM_STEPS_PER_ITER + 1;
            resampled_st_num_pts = resample_st(all_st_points, st_offset, st_length, STEP_SIZE,
                                               resampled_st_num_pts, resampled_st);

            // number of output directions
            if (num_st_in_avg == 0)
            {
                out_dirs_len = resampled_st_num_pts - 1;
            }

            float4 dir;
            float weight, cos_sim;
            for(int i_dir = 0; i_dir < min(out_dirs_len, resampled_st_num_pts -1); ++i_dir)
            {
                dir.xyz = normalize(resampled_st[1 + i_dir].xyz - resampled_st[i_dir].xyz);
                dir.w = 0.0f;  // make sure the 4th dimension stays zero

                cos_sim = num_st_in_avg > 0 ? dot(dir.xyz, normalize(out_dirs[i_dir].xyz)) : 0.0f;
                weight = num_st_in_avg > 0 ? (cos_sim > 0.0f ? cos_sim : 0.0f) : 1.0f;
                out_dirs[i_dir] += weight * dir;
            }
            ++num_st_in_avg;
        }
    }

    // normalize directions
    for(int i_dir = 0; i_dir < out_dirs_len; ++i_dir)
    {
        out_dirs[i_dir] = normalize(out_dirs[i_dir]);
    }

    return out_dirs_len;
}


int propagate_line(int num_strl_points, float4 curr_pos,
                   float4 last_dir, const size_t global_id,
                   __global const int* cell_ids,
                   __global const int* cell_st_counts,
                   __global const int* cell_st_offsets,
                   __global const int* cell_st_ids,
                   __global const int* all_st_offsets,
                   __global const float4* all_st_points,
                   uint* rng_state, __global float4* output_tracks)
{
    bool propagation_can_continue = true;
    while(num_strl_points < MAX_OUT_STRL_LEN && propagation_can_continue)
    {
        float4 next_dirs[NUM_STEPS_PER_ITER];
        const int num_next_dirs = get_next_directions(
            curr_pos, last_dir, cell_ids, cell_st_counts,
            cell_st_offsets, cell_st_ids, all_st_offsets,
            all_st_points, rng_state, next_dirs);

        if(num_next_dirs > 0)
        {
            for(int i_dir = 0; i_dir < num_next_dirs && num_strl_points < MAX_OUT_STRL_LEN; ++i_dir)
            {
                if(dot(next_dirs[i_dir].xyz, last_dir.xyz) > MIN_COS_ANGLE)
                {
                    curr_pos = curr_pos + STEP_SIZE * next_dirs[i_dir];
                    output_tracks[global_id*MAX_OUT_STRL_LEN+num_strl_points] = curr_pos;
                    last_dir.xyz = normalize(next_dirs[i_dir].xyz);
                    ++num_strl_points;
                }
                else
                {
                    // if there has been at last one valid direction, propagation can continue.
                    propagation_can_continue = i_dir > 0;
                    break;
                }
            }
        }
        else
        {
            propagation_can_continue = false;
        }
    }
    return num_strl_points;
}


bool get_init_direction(const float4 seed_pos, uint* rng_state,
                        const __global int* cell_ids, const __global int* cell_st_counts,
                        const __global int* cell_st_offsets, const __global int* cell_st_ids,
                        const __global int* all_st_offsets, const __global float4* all_st_points,
                        float4* init_dir)
{
    int neighbour_cells[MAX_SEARCH_NEIGHBOURS];
    const int num_neighbours = search_neighbours(seed_pos, neighbour_cells);

    int valid_st[MAX_SEARCH_NEIGHBOURS*MAX_DENSITY];
    const int num_valid_st = get_valid_trajectories(seed_pos,
        num_neighbours, neighbour_cells, cell_ids, cell_st_counts,
        cell_st_offsets, cell_st_ids, all_st_offsets, all_st_points,
        valid_st);

    // juste piocher dans nos short-tracks et prendre 1ere direction
    if(num_valid_st > 0)
    {
        rand_xorshift(rng_state);
        const float rand_val = (float)rng_state[0] / (float)UINT_MAX;
        const int rand_id = (int)(rand_val * num_valid_st);
        const int st_id = valid_st[rand_id];
        const int first_pt_id = all_st_offsets[st_id];
        init_dir[0].xyz = normalize(all_st_points[first_pt_id + 1].xyz -
                                    all_st_points[first_pt_id].xyz);
        return true;
    }
    return false;
}


void reverse_streamline(const int num_strl_points, const size_t global_id,
                        __global float4* output_tracks)
{
    for(int i = 0; i < (int)(num_strl_points/2); ++i)
    {
        const size_t head = global_id*MAX_OUT_STRL_LEN + i;
        const size_t tail = global_id*MAX_OUT_STRL_LEN+num_strl_points-1-i;
        const float4 temp_pt = output_tracks[global_id*MAX_OUT_STRL_LEN+i];
        output_tracks[head] = output_tracks[tail];
        output_tracks[tail] = temp_pt;
    }
}


__kernel void track_over_tracks(__global const int* cell_ids,
                                __global const int* cell_st_counts,
                                __global const int* cell_st_offsets,
                                __global const int* cell_st_ids,
                                __global const int* all_st_offsets,
                                __global const float4* all_st_points,
                                __global const float4* seed_points,
                                __global float4* output_tracks,
                                __global int* output_tracks_len)
{
    const size_t global_id = get_global_id(0);
    float4 position = seed_points[global_id];
    int num_strl_points = 0;
    uint rng_state = (uint)global_id + 1;

    float4 last_dir[1];
    bool found_init_dir = get_init_direction(
        position, &rng_state, cell_ids, cell_st_counts,
        cell_st_offsets, cell_st_ids, all_st_offsets,
        all_st_points, last_dir);

    if(found_init_dir)
    {
        // forward track
        output_tracks[global_id*MAX_OUT_STRL_LEN] = position;
        num_strl_points = propagate_line(
            1, seed_points[global_id], last_dir[0], global_id,
            cell_ids, cell_st_counts, cell_st_offsets, cell_st_ids,
            all_st_offsets, all_st_points, &rng_state, output_tracks);

        // if the number of point is smaller than MAX_OUT_STRL_LEN,
        // we can perform backward tracking.
        if(num_strl_points < MAX_OUT_STRL_LEN)
        {
            reverse_streamline(num_strl_points, global_id, output_tracks);

            // if the forward tracking yielded a streamline,
            // use it for initial direction in backward tracking
            if(num_strl_points > 1)
            {
                const float4 p0 = output_tracks[global_id*MAX_OUT_STRL_LEN+num_strl_points-2];
                const float4 p1 = output_tracks[global_id*MAX_OUT_STRL_LEN+num_strl_points-1];
                last_dir[0].xyz = normalize(p1.xyz - p0.xyz);
            }

            // backward track
            num_strl_points = propagate_line(
                num_strl_points, seed_points[global_id], last_dir[0], global_id,
                cell_ids, cell_st_counts, cell_st_offsets, cell_st_ids,
                all_st_offsets, all_st_points, &rng_state, output_tracks);
        }
    }
    output_tracks_len[global_id] = num_strl_points;
}
