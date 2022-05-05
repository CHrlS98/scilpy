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
#define MAX_SEARCH_NEIGHBOURS 0 // search_radius / edge_length + 1 as type int
#define XMAX 0
#define YMAX 0
#define ZMAX 0

// everything above next line will be removed
//$TRIMABOVE$
const int4 CELLS_GRID_DIMS = {XMAX, YMAX, ZMAX, 0};

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
    const int3 cell_id_3dimensional = {(int)point_in_cells_space.x,
                                       (int)point_in_cells_space.y,
                                       (int)point_in_cells_space.z};
    // ravel 3d index in 1d
    return cell_id_3dimensional.z * CELLS_GRID_DIMS.y * CELLS_GRID_DIMS.x +
           cell_id_3dimensional.y * CELLS_GRID_DIMS.x + cell_id_3dimensional.x;
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

// TODO: VERIFY WELL CODED
int search_neighbours(const float4 pos, const int2 x_bounds,
                      const int2 y_bounds, const int2 z_bounds,
                      int* neighbours)
{
    int num_neighbours = 0;
    for(float x_offset = -x_bounds.x; x_offset <= x_bounds.y; x_offset += EDGE_LENGTH)
    {
        for(float y_offset = -y_bounds.x; y_offset <= y_bounds.y; y_offset += EDGE_LENGTH)
        {
            for(float z_offset = -z_bounds.x; z_offset <= z_bounds.y; z_offset += EDGE_LENGTH)
            {
                // new position is inside image range
                const float4 new_position = {x_offset, y_offset, z_offset, 1.0f};
                const int cell_id = map_to_cell_id(new_position);
                if(!contains_unsorted(cell_id, neighbours, num_neighbours))
                {
                    neighbours[num_neighbours++] = cell_id;
                }
            }
        }
    }
    return num_neighbours;
}


void get_search_boundaries(const float4 center, float2* x_bounds,
                           float2* y_bounds, float2* z_bounds)
{
    // new position is inside image range
    x_bounds[0].x = clamp(center.x - SEARCH_RADIUS, 0.0f, (float)CELLS_GRID_DIMS.x);
    x_bounds[0].y = clamp(center.x + SEARCH_RADIUS, 0.0f, (float)CELLS_GRID_DIMS.x);

    y_bounds[0].x = clamp(center.y - SEARCH_RADIUS, 0.0f, (float)CELLS_GRID_DIMS.y);
    y_bounds[0].y = clamp(center.y + SEARCH_RADIUS, 0.0f, (float)CELLS_GRID_DIMS.y);

    z_bounds[0].x = clamp(center.z - SEARCH_RADIUS, 0.0f, (float)CELLS_GRID_DIMS.z);
    z_bounds[0].y = clamp(center.z + SEARCH_RADIUS, 0.0f, (float)CELLS_GRID_DIMS.z);
}


__kernel void track_over_tracks(__global const int* cell_ids,
                                __global const int* cell_strl_counts,
                                __global const int* cell_strl_offsets,
                                __global const int* cell_strl_ids,
                                __global const int* strl_lengths,
                                __global const int* strl_offsets,
                                __global const float4* strl_points,
                                __global const float4* seed_points,
                                __global float4* output_tracks)
{
    // TODO
    // 1. Utiliser get_global_id(0) pour trouver l'index de la seed;
    // ... Me donne aussi le start index dans output_tracks;
    // 2. Trouver toutes les streamlines uniques qui sont a proximite de
    // ... ma seed position (map_to_cell_id, dichotomic_search_cell);
    // 3. 
    const float4 seed_point = seed_points[get_global_id(0)];

    // On trouve les bornes de la région à chercher
    float2 x_bounds[1];
    float2 y_bounds[1];
    float2 z_bounds[1];
    get_search_boundaries(seed_point, x_bounds, y_bounds, z_bounds);

    int neighbour_cells[MAX_SEARCH_NEIGHBOURS];
}