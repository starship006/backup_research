from imports import *
from beartype import beartype as typechecker

@jaxtyped
@typechecker
def zero_ablation_hook(
    attn_result: Union[Float[Tensor, "batch seq n_heads d_model"], Float[Tensor, "batch seq d_model"]],
    hook: HookPoint,
    head_index_to_ablate: int = -1000,
) -> Union[Float[Tensor, "batch seq n_heads d_model"], Float[Tensor, "batch seq d_model"]]:
    """
    zero ablates a head or mlp layer across all batch x positions
    """

    if len(attn_result.shape) == 3:
        attn_result[:] = torch.zeros(attn_result.shape)
    else:
        # attention head
        attn_result[:, :, head_index_to_ablate, :] = torch.zeros(attn_result[:, :, head_index_to_ablate, :].shape)
    return attn_result




def overwrite_activation_hook(
        original_activation,
        hook,
        what_to_overwrite_with
    ):
        assert original_activation.shape == what_to_overwrite_with.shape
        return what_to_overwrite_with
    
    


def add_and_replace_hook(
    original_activation,
    hook,
    what_to_overwrite_with,
    added_component_in_list
):
    assert original_activation.shape == what_to_overwrite_with.shape == added_component_in_list[0].shape
    
    original_activation[:] = what_to_overwrite_with + added_component_in_list[0]


def store_item_in_tensor(item: Tensor, hook, tensor: Tensor):
    assert item.shape == tensor.shape
    tensor[:] = item
    return tensor