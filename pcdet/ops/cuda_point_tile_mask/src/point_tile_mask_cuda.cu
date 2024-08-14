#include <cmath>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

//#define POINTS_PER_THREAD 1

template <typename scalar_t>
using one_dim_pa32 = torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits>;

template <typename scalar_t>
__global__ void point_tile_mask_cuda_kernel(
        const one_dim_pa32<scalar_t>  tile_coords,
        const one_dim_pa32<scalar_t>  chosen_tile_coords,
        one_dim_pa32<bool> mask) {
  auto chosen_tile_coords_sz = chosen_tile_coords.size(0);
  
  // With shared memory, the time reduced to 0.25 ms from 0.37 ms
  extern __shared__ __align__(sizeof(scalar_t)) unsigned char smem[];
  scalar_t *chsn_tile_coords= reinterpret_cast<scalar_t *>(smem);
  
  if(threadIdx.x < chosen_tile_coords_sz)
      chsn_tile_coords[threadIdx.x] = chosen_tile_coords[threadIdx.x];
  __syncthreads();
  
  // blockIdx.x is the block id
  // blockDim.x is the number of threads in a block
  // threadIdx.x is the thread id in the block
  auto tile_coord_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(tile_coord_idx < tile_coords.size(0)){
    auto tile_coord = tile_coords[tile_coord_idx];
    bool includes=false;
    // Check if tile_coord is included in chosen tile coords or not
    for(auto i=0; i < chosen_tile_coords_sz; ++i){
      includes |= (tile_coord == chsn_tile_coords[i]);
    }
    mask[tile_coord_idx] = includes;
  }
}

torch::Tensor point_tile_mask(
        torch::Tensor tile_coords, // [num_points] x*w+y notation
        torch::Tensor chosen_tile_coords // [num_chosen_tiles] x*w+y notation
)
{

  auto threads_per_block = 256;
  while(threads_per_block < chosen_tile_coords.size(0))
    threads_per_block += 128;

  const auto num_blocks = std::ceil((double)tile_coords.size(0) / threads_per_block);
 
  auto tensor_options = torch::TensorOptions()
      .layout(torch::kStrided)
      .dtype(torch::kBool) // Bool
      .device(tile_coords.device().type())
      .requires_grad(false);

  torch::Tensor mask = torch::empty({tile_coords.size(0)}, tensor_options);

  AT_DISPATCH_INTEGRAL_TYPES(tile_coords.type(), "point_tile_mask_cuda", ([&] {
    auto sh_mem_size = chosen_tile_coords.size(0)*sizeof(scalar_t);
    point_tile_mask_cuda_kernel<scalar_t><<<num_blocks, threads_per_block, sh_mem_size>>>(
      tile_coords.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
      chosen_tile_coords.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
      mask.packed_accessor32<bool,1,torch::RestrictPtrTraits>());
  }));

  return mask;
}
