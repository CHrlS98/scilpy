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
#define MAX_DENSITY 0 // maximum number of streamlines per cell
#define MAX_SEARCH_NEIGHBOURS 0 // search_radius / edge_length + 1 as type int, to the power of 3
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


int search_neighbours(const float4 pos, int* neighbours)
{
    float2 x_bounds[1];
    float2 y_bounds[1];
    float2 z_bounds[1];
    get_search_boundaries(pos, x_bounds, y_bounds, z_bounds);

    int num_neighbours = 0;
    for(float x_offset = -x_bounds[0].x; x_offset <= x_bounds[0].y; x_offset += EDGE_LENGTH)
    {
        for(float y_offset = -y_bounds[0].x; y_offset <= y_bounds[0].y; y_offset += EDGE_LENGTH)
        {
            for(float z_offset = -z_bounds[0].x; z_offset <= z_bounds[0].y; z_offset += EDGE_LENGTH)
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


int get_valid_trajectories(const float4 current_position, const int num_neighbours, const int* neighbour_cells,
                          __global const int* cell_ids, __global const int* cell_st_counts,
                          __global const int* cell_st_offsets, __global const int* cell_st_ids,
                          __global const int* all_st_lengths, __global const int* all_st_offsets,
                          __global const float4* all_st_points, int* valid_st, int* closest_pts)
{
    // maximum number of short-tracks is MAX_SEARCH_NEIGHBOURS*MAX_DENSITY
    int unique_st[MAX_SEARCH_NEIGHBOURS*MAX_DENSITY];
    int num_unique_st = 0;

    // passer au travers de chaque cellule et ajouter les short-tracks ids sans duplicats
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

    // unique_streamlines peuvent être trop loin pour être valides
    // on doit faire le tri en passant à travers chaque streamline.
    int num_valid_st = 0;
    for(int i  = 0; i < num_unique_st; ++i)
    {
        const int st_id = unique_st[i];
        const int st_length = all_st_lengths[st_id];
        const int st_offset = all_st_offsets[st_id];

        float min_distance = SEARCH_RADIUS + 1.0f;
        int closest_pt_id = -1;
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
        if(closest_pt_id >= 0)
        {
            valid_st[num_valid_st] = st_id;
            closest_pts[num_valid_st] = closest_pt_id;
            ++num_valid_st;
        }
    }
    return num_valid_st;
}


float4 get_next_direction(const float4 current_position, const int num_neighbours, const int* neighbour_cells,
                          __global const int* cell_ids, __global const int* cell_st_counts,
                          __global const int* cell_st_offsets, __global const int* cell_st_ids,
                          __global const int* st_lengths, __global const int* st_offsets,
                          __global const float4* st_points)
{
    int valid_st[MAX_SEARCH_NEIGHBOURS*MAX_DENSITY];
    int closest_pts[MAX_SEARCH_NEIGHBOURS*MAX_DENSITY];
    const int num_valid_st = get_valid_trajectories(current_position, num_neighbours, neighbour_cells,
                                                    cell_ids, cell_st_counts, cell_st_offsets, cell_st_ids,
                                                    st_lengths, st_offsets, st_points, valid_st, closest_pts);
    // valid_streamlines now contains tous les indices des short-tracks à l'intérieur du search radius
    for(int i = 0; i < num_valid_st; ++i)
    {
        const int valid_st_id = valid_st[i];
    }
    return (float4)(0.0f, 0.0f, 0.0f, 0.0f);
}


// Naming convention: 'st' means 'short-tracks'. Short-tracks are used as input,
//                    the input is 'tracks' or 'strl' (streamlines)
__kernel void track_over_tracks(__global const int* cell_ids, // liste des cellules contenant des streamlines
                                __global const int* cell_st_counts, // liste du nombre de short-tracks contenue dans chaque cellule
                                __global const int* cell_st_offsets, // liste du decalage dans le tableau des short-tracks ids pour chaque cellule
                                __global const int* cell_st_ids, // tableau des index des short-tracks contenue dans chaque cellule
                                __global const int* all_st_lengths, // tableau de longueur n_short_tracks qui indique le nombre de points de chaque shorttrack
                                __global const int* all_st_offsets, // tableau de longueur n_short_tracks qui indique l'index où commence la n-ieme shorttrack
                                __global const float4* all_st_points, // tous les points de toutes les streamlines
                                __global const float4* seed_points,
                                __global float4* output_tracks)
{
    const float4 seed_point = seed_points[get_global_id(0)];

    int neighbour_cells[MAX_SEARCH_NEIGHBOURS];
    const int num_neighbours = search_neighbours(seed_point, neighbour_cells);
    // neighbour_cells est une liste qui contient num_neighbours cell_ids
    // chaque cell ID est associée à des streamlines qui la traversent.

    // tracking loop qui fait des steps dans mon short-tracks field.
    // pas besoin de mask puisque mes short-tracks ont deja ete filtrees avec
    // un masque.

    // commencer par tester que je peux retrouver l'ensemble de short-tracks le plus proche d'un certain point

    // short-tracks pourraient etre compressees avec une regression polynomiales
}
