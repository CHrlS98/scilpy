/*
OpenCL kernel code for tracking over short-tracks tractograms.

NOTES:
* streamlines have maximum length;
* streamlines are forward- and backward-tracked;
*/

// placeholder values for compiler definitions
#define NUM_CELLS 0
#define SEARCH_RADIUS 0.0f
#define EDGE_LENGTH 0.0f
#define MAX_DENSITY 0
#define MAX_SEARCH_NEIGHBOURS 0
#define CELLS_XMAX 0
#define CELLS_YMAX 0
#define CELLS_ZMAX 0
#define MAX_STRL_LEN 0
#define MIN_COS_ANGLE 0.0f
#define MIN_COS_ANGLE_INIT 0.0f
#define STEP_SIZE 0.0f

// everything above next line will be removed
//$TRIMABOVE$
const float FLOAT_EPS = 0.0001f;
const int4 CELLS_GRID_DIMS = {CELLS_XMAX, CELLS_YMAX, CELLS_ZMAX, 0};

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
    const float4 point_in_cells_space = point / EDGE_LENGTH;

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


void get_search_boundaries(const float4 center, float2* x_bounds,
                           float2* y_bounds, float2* z_bounds)
{
    // new position is inside image range
    x_bounds[0].x = center.x - SEARCH_RADIUS;
    x_bounds[0].y = center.x + SEARCH_RADIUS;

    y_bounds[0].x = center.y - SEARCH_RADIUS;
    y_bounds[0].y = center.y + SEARCH_RADIUS;

    z_bounds[0].x = center.z - SEARCH_RADIUS;
    z_bounds[0].y = center.z + SEARCH_RADIUS;
}


int search_neighbours(const float4 pos, int* neighbours)
{
    float2 x_bounds[1];
    float2 y_bounds[1];
    float2 z_bounds[1];
    get_search_boundaries(pos, x_bounds, y_bounds, z_bounds);

    int num_neighbours = 0;
    for(float x_offset = x_bounds[0].x; x_offset < x_bounds[0].y + FLOAT_EPS; x_offset += EDGE_LENGTH)
    {
        for(float y_offset = y_bounds[0].x; y_offset < y_bounds[0].y + FLOAT_EPS; y_offset += EDGE_LENGTH)
        {
            for(float z_offset = z_bounds[0].x; z_offset < z_bounds[0].y + FLOAT_EPS; z_offset += EDGE_LENGTH)
            {
                // new position is inside image range
                const float4 new_position = {x_offset, y_offset, z_offset, 1.0f};
                const int cell_id = map_to_cell_id(new_position);
                if(cell_id != -1) // outside image bounds
                {
                    if(!contains_unsorted(cell_id, neighbours, num_neighbours))
                    {
                        neighbours[num_neighbours++] = cell_id;
                    }
                }
            }
        }
    }
    return num_neighbours;
}

int get_unique_trajectories(const int num_neighbours, const int* neighbour_cells,
                            __global const int* cell_ids, __global const int* cell_st_counts,
                            __global const int* cell_st_offsets, __global const int* cell_st_ids,
                            int* unique_st)
{
    int num_unique_st = 0;

    // go through all neighbour cells and add short-track ids, without duplicates
    for(int n_id = 0; n_id < num_neighbours; ++n_id)
    {
        const int neigh_cell_id = neighbour_cells[n_id];
        const int current_cell_index = dichotomic_search_cell(neigh_cell_id, cell_ids);
        if(current_cell_index > -1)
        {
            // this neighbour contains streamlines
            const int st_count = cell_st_counts[current_cell_index];
            const int st_offset = cell_st_offsets[current_cell_index];

            // iterate through all streamlines in current cell
            for(int st_i = st_offset; st_i < st_offset + st_count; ++st_i)
            {
                const int st_id = cell_st_ids[st_i];
                if(!contains_unsorted(st_id, unique_st, num_unique_st))
                {
                    unique_st[num_unique_st++] = st_id;
                }
            }
        }
    }
    return num_unique_st;
}


int get_valid_trajectories(const float4 current_position, const float4 prev_dir, const int num_neighbours,
                           const int* neighbour_cells, __global const int* cell_ids,
                           __global const int* cell_st_counts, __global const int* cell_st_offsets,
                           __global const int* cell_st_ids, __global const int* all_st_lengths,
                           __global const int* all_st_offsets, __global const float4* all_st_points,
                           float4* next_dir)
{
    int unique_st[MAX_SEARCH_NEIGHBOURS*MAX_DENSITY];
    int num_unique_st = get_unique_trajectories(
        num_neighbours, neighbour_cells, cell_ids, cell_st_counts,
        cell_st_offsets, cell_st_ids, unique_st);

    // The candidate short-tracks may be outside the search radius,
    // so we need to compute the distance between the current position
    // and each short-tracks.
    int num_valid_st = 0;
    float4 dir_i;
    for(int i  = 0; i < num_unique_st; ++i)
    {
        const int st_id = unique_st[i];
        const int st_length = all_st_lengths[st_id];
        const int st_offset = all_st_offsets[st_id];

        float min_distance = SEARCH_RADIUS;
        int closest_pt_id = -1;
        // TODO: Je veux:
        // 1. que ma plus petite MEAN DISTANCE soit inferieure a mon search radius!!
        for(int point_id = st_offset; point_id < st_offset + st_length; ++point_id)
        {
            // to be valid, a streamline must have at least
            // one point inside the search_radius
            const float4 current_st_point = all_st_points[point_id];
            const float dist_to_curr_position = distance(current_st_point.xyz, current_position.xyz);
            if(dist_to_curr_position < min_distance)
            {
                min_distance = dist_to_curr_position;
                closest_pt_id = point_id;
            }
        }

        // if short-track is valid
        if(closest_pt_id >= 0)
        {
            dir_i = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
            const int first_pt_id = all_st_offsets[st_id];
            const int last_pt_id = first_pt_id + all_st_lengths[st_id];
            if(closest_pt_id > first_pt_id)
            {
                // not the first point of streamline
                const float4 p0 = all_st_points[closest_pt_id - 1];
                const float4 p1 = all_st_points[closest_pt_id];
                dir_i.xyz += (p1 - p0).xyz;
            }
            if(closest_pt_id < last_pt_id - 1)
            {
                // not the last point of streamline
                const float4 p0 = all_st_points[closest_pt_id];
                const float4 p1 = all_st_points[closest_pt_id + 1];
                dir_i.xyz += (p1 - p0).xyz;
            }
            if(dot(normalize(dir_i.xyz), prev_dir.xyz) < 0.0f)
            {
                dir_i.xyz = -dir_i.xyz;
            }

            // deviation angle test
            const float deviation = dot(normalize(dir_i.xyz), prev_dir.xyz);
            if(deviation > MIN_COS_ANGLE)
            {
                next_dir[0].xyz += deviation * dir_i.xyz;
            }
            ++num_valid_st;
        }
    }
    return num_valid_st;
}


bool get_next_direction(const float4 curr_pos, const float4 prev_dir,
                        __global const int* cell_ids,
                        __global const int* cell_st_counts,
                        __global const int* cell_st_offsets,
                        __global const int* cell_st_ids,
                        __global const int* all_st_lengths,
                        __global const int* all_st_offsets,
                        __global const float4* all_st_points,
                        int* neighbour_cells, int* valid_st,
                        int* closest_pts, float4* next_dir)
{
    const int num_neighbours = search_neighbours(curr_pos, neighbour_cells);
    next_dir[0] = (float4)(0.0f, 0.0f, 0.0f, 1.0f);
    const int num_valid_st = get_valid_trajectories(
        curr_pos, prev_dir, num_neighbours, neighbour_cells,
        cell_ids, cell_st_counts, cell_st_offsets, cell_st_ids,
        all_st_lengths, all_st_offsets, all_st_points, next_dir);

    if(num_valid_st > 0)
    {
        // normalize direction
        next_dir[0].xyz = normalize(next_dir[0].xyz);
        return true;
    }
    return false;
}


int propagate_line(int num_strl_points, float4 curr_pos,
                   float4 last_dir, const size_t global_id,
                   __global const int* cell_ids,
                   __global const int* cell_st_counts,
                   __global const int* cell_st_offsets,
                   __global const int* cell_st_ids,
                   __global const int* all_st_lengths,
                   __global const int* all_st_offsets,
                   __global const float4* all_st_points,
                   __global float4* output_tracks)
{
    int neighbour_cells[MAX_SEARCH_NEIGHBOURS];
    int valid_st[MAX_SEARCH_NEIGHBOURS*MAX_DENSITY];
    int closest_pts[MAX_SEARCH_NEIGHBOURS*MAX_DENSITY];

    bool propagation_can_continue = true;
    while(num_strl_points < MAX_STRL_LEN && propagation_can_continue)
    {
        float4 next_dir[1];
        const bool is_valid = get_next_direction(
            curr_pos, last_dir, cell_ids, cell_st_counts,
            cell_st_offsets, cell_st_ids, all_st_lengths,
            all_st_offsets, all_st_points, neighbour_cells,
            valid_st, closest_pts, next_dir);

        if(is_valid)
        {
            curr_pos = curr_pos + STEP_SIZE * next_dir[0];
            last_dir = next_dir[0];
            output_tracks[global_id*MAX_STRL_LEN+num_strl_points] = curr_pos;
            ++num_strl_points;
        }
        else
        {
            propagation_can_continue = false;
        }
    }
    return num_strl_points;
}


bool get_init_direction(const float4 seed_pos,
                        __global const int* cell_ids,
                        __global const int* cell_st_counts,
                        __global const int* cell_st_offsets,
                        __global const int* cell_st_ids,
                        __global const int* all_st_lengths,
                        __global const int* all_st_offsets,
                        __global const float4* all_st_points,
                        float4* init_dir)
{
    int neighbour_cells[MAX_SEARCH_NEIGHBOURS];

    const int num_neighbours = search_neighbours(seed_pos, neighbour_cells);

    int unique_st[MAX_SEARCH_NEIGHBOURS*MAX_DENSITY];
    int num_unique_st = get_unique_trajectories(
        num_neighbours, neighbour_cells, cell_ids, cell_st_counts,
        cell_st_offsets, cell_st_ids, unique_st);

    int num_valid_st = 0;
    float4 dir_i;
    float4 direction = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    for(int i  = 0; i < num_unique_st; ++i)
    {
        const int st_id = unique_st[i];
        const int st_length = all_st_lengths[st_id];
        const int st_offset = all_st_offsets[st_id];

        float min_distance = SEARCH_RADIUS;
        int closest_pt_id = -1;
        for(int point_id = st_offset; point_id < st_offset + st_length; ++point_id)
        {
            // to be valid, a streamline must have at least
            // one point inside the search_radius
            const float4 current_st_point = all_st_points[point_id];
            const float dist_to_curr_position = distance(current_st_point.xyz, seed_pos.xyz);
            if(dist_to_curr_position < min_distance)
            {
                min_distance = dist_to_curr_position;
                closest_pt_id = point_id;
            }
        }
        if(closest_pt_id >= 0)
        {
            dir_i = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
            const int first_pt_id = all_st_offsets[st_id];
            const int last_pt_id = first_pt_id + all_st_lengths[st_id];
            if(closest_pt_id > first_pt_id)
            {
                // not the first point of streamline
                const float4 p0 = all_st_points[closest_pt_id - 1];
                const float4 p1 = all_st_points[closest_pt_id];
                dir_i.xyz += (p1 - p0).xyz;
            }
            if(closest_pt_id < last_pt_id - 1)
            {
                // not the last point of streamline
                const float4 p0 = all_st_points[closest_pt_id];
                const float4 p1 = all_st_points[closest_pt_id + 1];
                dir_i.xyz += (p1 - p0).xyz;
            }

            if(num_valid_st == 0)
            {
                direction = dir_i;
            }
            else if(dot(normalize(dir_i.xyz), normalize(direction.xyz)) < 0.0f)
            {
                dir_i.xyz = -dir_i.xyz;
            }
            // deviation angle test
            if(dot(normalize(dir_i.xyz), normalize(direction.xyz)) > MIN_COS_ANGLE_INIT)
            {
                direction.xyz += dir_i.xyz;
                num_valid_st += 1;
            }
        }
    }

    init_dir[0].xyz = direction.xyz;
    init_dir[0].w = 1.0f;
    return num_valid_st > 0;
}


void reverse_streamline(const int num_strl_points, const size_t global_id,
                        __global float4* output_tracks)
{
    for(int i = 0; i < (int)(num_strl_points/2); ++i)
    {
        const size_t head = global_id*MAX_STRL_LEN + i;
        const size_t tail = global_id*MAX_STRL_LEN+num_strl_points-1-i;
        const float4 temp_pt = output_tracks[global_id*MAX_STRL_LEN+i];
        output_tracks[head] = output_tracks[tail];
        output_tracks[tail] = temp_pt;
    }
}


// Naming convention: 'st' means 'short-tracks'. Short-tracks are used as input,
//                    the output is 'tracks' or 'strl' (streamlines)
__kernel void track_over_tracks(__global const int* cell_ids,
                                __global const int* cell_st_counts,
                                __global const int* cell_st_offsets,
                                __global const int* cell_st_ids,
                                __global const int* all_st_lengths,
                                __global const int* all_st_offsets,
                                __global const float4* all_st_points,
                                __global const float4* seed_points,
                                __global float4* output_tracks,
                                __global int* output_tracks_len)
{
    const size_t global_id = get_global_id(0);
    float4 position = seed_points[global_id];
    int num_strl_points = 0;

    float4 last_dir[1];
    bool found_init_dir = get_init_direction(
        position, cell_ids, cell_st_counts, cell_st_offsets, cell_st_ids,
        all_st_lengths, all_st_offsets, all_st_points, last_dir);

    if(found_init_dir)
    {
        // forward track
        output_tracks[global_id*MAX_STRL_LEN] = position;
        num_strl_points = propagate_line(
            1, seed_points[global_id], last_dir[0], global_id,
            cell_ids, cell_st_counts, cell_st_offsets, cell_st_ids,
            all_st_lengths, all_st_offsets, all_st_points, output_tracks);

        // if the number of point is smaller than MAX_STRL_LEN,
        // we can perform backward tracking.
        if(num_strl_points < MAX_STRL_LEN)
        {
            reverse_streamline(num_strl_points, global_id, output_tracks);

            // if the forward tracking yielded a streamline,
            // use it for initial direction in backward tracking
            if(num_strl_points > 1)
            {
                const float4 p0 = output_tracks[global_id*MAX_STRL_LEN+num_strl_points-2];
                const float4 p1 = output_tracks[global_id*MAX_STRL_LEN+num_strl_points-1];
                last_dir[0].xyz = normalize(p1.xyz - p0.xyz);
            }

            // backward track
            num_strl_points = propagate_line(
                num_strl_points, seed_points[global_id], last_dir[0], global_id,
                cell_ids, cell_st_counts, cell_st_offsets, cell_st_ids,
                all_st_lengths, all_st_offsets, all_st_points, output_tracks);
        }
    }
    output_tracks_len[global_id] = num_strl_points;
}
