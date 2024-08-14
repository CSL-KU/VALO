#include <torch/extension.h>
#include <pybind11/pybind11.h>

#include <iostream>
#include <utility>
#include <vector>
#include <map>

using namespace torch::indexing;
namespace py = pybind11;

// Returns the projected pred_boxes and the mask to get them
torch::Tensor project_past_detections(
        torch::Tensor pred_boxes, // [num_objects, 9]
        torch::Tensor past_pose_indexes, // [num_objects]
        torch::Tensor past_poses, // [num_past_poses, 14]
        torch::Tensor cur_pose, // [14]
        torch::Tensor past_timestamps, // [num_past_poses]
        long cur_timestamp // [1]
);


//std::vector<std::map<std::string, torch::Tensor>> split_projections(
//        torch::Tensor pred_boxes, // [num_objects, 9], fp_type
//        torch::Tensor pred_scores,
//        torch::Tensor pred_labels,
//	torch::Tensor cls_id_to_det_head_idx_map,
//	int num_det_heads
//)
//{
//	std::vector<int> label_counts(num_det_heads, 0);
//	std::vector<int> det_head_mappings(pred_labels.size(0));
//	auto pred_labels_a = pred_labels.accessor<long,1>();
//	auto map_a = cls_id_to_det_head_idx_map.accessor<int,1>();
//
//	for(auto i=0; i<pred_labels_a.size(0); ++i){
//		det_head_mappings[i] = map_a[pred_labels_a[i]];
//		++label_counts[det_head_mappings[i]];
//	}
//
//	std::vector<torch::Tensor> pred_boxes_arr, pred_scores_arr, pred_labels_arr;
//	auto tensor_options = torch::TensorOptions()
//		.layout(torch::kStrided)
//		.device(pred_boxes.device().type())
//		.requires_grad(false);
//
//	for(auto i=0; i<num_det_heads; ++i){
//		pred_boxes_arr.push_back(torch::empty({label_counts[i], pred_boxes.size(1)},
//				tensor_options.dtype(pred_boxes.dtype())));
//		pred_scores_arr.push_back(torch::empty({label_counts[i]},
//				tensor_options.dtype(pred_scores.dtype())));
//		pred_labels_arr.push_back(torch::empty({label_counts[i]},
//				tensor_options.dtype(pred_labels.dtype())));
//	}
//
//	for(auto i=0; i<pred_labels_a.size(0); i++){
//		auto m = det_head_mappings[i];
//		auto j = --label_counts[m];
//		pred_boxes_arr[m].index_put_({j}, pred_boxes.index({i}));
//		pred_scores_arr[m].index_put_({j}, pred_scores.index({i}));
//		pred_labels_arr[m].index_put_({j}, pred_labels.index({i}));
//	}
//
//	std::vector<std::map<std::string, torch::Tensor>> proj_dicts(num_det_heads);
//	for(auto i=0; i<num_det_heads; ++i){
//		std::map<std::string, torch::Tensor> pdict;
//		pdict["pred_boxes"] = pred_boxes_arr[i].to(torch::kCUDA);
//		pdict["pred_scores"] = pred_scores_arr[i].to(torch::kCUDA);
//		pdict["pred_labels"] = pred_labels_arr[i].to(torch::kCUDA);
//		proj_dicts[i] = std::move(pdict);
//	}
//	return proj_dicts;
//}


std::vector<std::map<std::string, torch::Tensor>> split_projections(
        torch::Tensor pred_boxes, // [num_objects, 9], fp_type
        torch::Tensor pred_scores,
        torch::Tensor pred_labels,
	torch::Tensor cls_id_to_det_head_idx_map,
	int num_det_heads,
	bool move_to_gpu
)
{
	using namespace torch::indexing;
	torch::Tensor det_head_mappings = cls_id_to_det_head_idx_map.index({pred_labels});

	std::vector<std::map<std::string, torch::Tensor>> proj_dicts(num_det_heads);
	auto pred_merged = torch::cat({pred_boxes, pred_scores.unsqueeze(-1),
			pred_labels.to(torch::kFloat32).unsqueeze(-1)}, -1);
	for(auto i=0; i<num_det_heads; ++i){
		std::map<std::string, torch::Tensor> pdict;

		auto pred_masked = pred_merged.index({(det_head_mappings == i)});
		auto pb = pred_masked.index({Slice(), Slice(None, -2)});
		auto ps = pred_masked.index({Slice(), -2});
		auto pl = pred_masked.index({Slice(), -1}).to(torch::kLong);
		pdict["pred_boxes"] = (move_to_gpu ? pb.to(torch::kCUDA) : pb);
		pdict["pred_scores"] = (move_to_gpu ? ps.to(torch::kCUDA) : ps);
		pdict["pred_labels"] = (move_to_gpu ? pl.to(torch::kCUDA) : pl);
		proj_dicts[i] = std::move(pdict);
	}
	return proj_dicts;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("project_past_detections", &project_past_detections, "Detection projector CUDA");
    m.def("split_projections", &split_projections, "Detection splitter");
}
