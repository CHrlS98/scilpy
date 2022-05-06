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
                          __global const int* cell_ids, __global const int* cell_strl_counts,
                          __global const int* cell_strl_offsets, __global const int* cell_strl_ids,
                          __global const int* all_strl_lengths, __global const int* all_strl_offsets,
                          __global const float4* all_strl_points, int* valid_streamlines)
{
    // maximum number of streamlines is MAX_SEARCH_NEIGHBOURS*MAX_DENSITY
    int unique_streamlines[MAX_SEARCH_NEIGHBOURS*MAX_DENSITY];
    int num_unique_strls = 0;

    // passer au travers de chaque cellule et ajouter les streamlines ids sans duplicats
    for(int n_id = 0; n_id < num_neighbours; ++n_id)
    {
        const int neigh_cell_id = neighbour_cells[n_id];
        const int current_cell_index = dichotomic_search_cell(neigh_cell_id, cell_ids);
        if(current_cell_index > -1)
        {
            // this neighbour contains streamlines
            const int strl_count = cell_strl_counts[current_cell_index];
            const int strl_offset = cell_strl_offsets[current_cell_index];

            // iterate through all streamlines in current cell
            for(int strl_i = strl_offset; strl_i < strl_offset + strl_count; ++strl_i)
            {
                const int strl_id = cell_strl_ids[strl_i];
                if(!contains_unsorted(strl_id, unique_streamlines, num_unique_strls))
                {
                    unique_streamlines[num_unique_strls++] = strl_id;
                }
            }
        }
    }

    // unique_streamlines peuvent être trop loin pour être valides
    // on doit faire le tri en passant à travers chaque streamline.
    int num_valid_streamlines = 0;
    for(int i  = 0; i < num_unique_strls; ++i)
    {
        const int strl_id = unique_streamlines[i];
        const int strl_length = all_strl_lengths[strl_id];
        const int strl_offset = all_strl_offsets[strl_id];

        bool is_inside_radius = false;
        for(int point_id = strl_offset;
            point_id < strl_offset + strl_length && !is_inside_radius;
            ++point_id)
        {
            // to be valid, a streamline must have at least
            // one point inside the search_radius
            const float4 current_strl_point = all_strl_points[point_id];
            is_inside_radius =
                distance(current_strl_point.xyz, current_position.xyz) < SEARCH_RADIUS;
        }
        if(is_inside_radius)
        {
            valid_streamlines[num_valid_streamlines++] = strl_id;
        }
    }
    return num_valid_streamlines;
}


float4 get_next_direction(const float4 current_position, const int num_neighbours, const int* neighbour_cells,
                          __global const int* cell_ids, __global const int* cell_strl_counts,
                          __global const int* cell_strl_offsets, __global const int* cell_strl_ids,
                          __global const int* strl_lengths, __global const int* strl_offsets,
                          __global const float4* strl_points)
{
    int valid_streamlines[MAX_SEARCH_NEIGHBOURS*MAX_DENSITY];
    get_valid_trajectories(current_position, num_neighbours, neighbour_cells, cell_ids,
                           cell_strl_counts, cell_strl_offsets, cell_strl_ids, strl_lengths,
                           strl_offsets, strl_points, valid_streamlines);
    // valid_streamlines now contains tous les indices des short-tracks à l'intérieur du search radius
}


// TODO: Better naming. Probablement que le output est des
//       streamlines, mais le input est plutôt des short-tracks.
//
// Il faudrait que ce soit moins mêlant.
__kernel void track_over_tracks(__global const int* cell_ids, // liste des cellules contenant des streamlines
                                __global const int* cell_strl_counts, // liste du nombre de streamlines contenue dans chaque cellule
                                __global const int* cell_strl_offsets, // liste du decalage dans le tableau des streamlines ids pour chaque cellule
                                __global const int* cell_strl_ids, // tableau des index des streamlines contenue dans chaque cellule
                                __global const int* strl_lengths, // tableau de longueur n_streamlines qui indique le nombre de points de chaque streamline
                                __global const int* strl_offsets, // tableau de longueur n_streamlines qui indique l'index où commence la n-ieme streamline
                                __global const float4* strl_points, // tous les points de toutes les streamlines
                                __global const float4* seed_points,
                                __global float4* output_tracks)
{
    const float4 seed_point = seed_points[get_global_id(0)];

    int neighbour_cells[MAX_SEARCH_NEIGHBOURS];
    const int num_neighbours = search_neighbours(seed_point, neighbour_cells);
    // neighbour_cells est une liste qui contient num_neighbours cell_ids
    // chaque cell ID est associée à des streamlines qui la traversent.

    // tracking loop qui fait des steps dans mon short-tracks field.
}
