# src/models/vision.py

from typing import Callable
import torch.nn as nn
import torchvision


def get_resnet(name: str, weights=None, **kwargs) -> nn.Module:
    """
    Helper to create a standard ResNet backbone from torchvision.

    Args:
        name:    "resnet18", "resnet34", "resnet50", ...
        weights: e.g. torchvision.models.ResNet18_Weights.IMAGENET1K_V1, or None.
                 (원 노트북에서는 문자열을 썼지만, 최신 torchvision에선 Enum 사용 권장)

    Returns:
        ResNet backbone with the final fully connected layer removed (fc = Identity).
    """
    # torchvision.models 내에서 이름으로 생성 함수 가져오기
    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)

    # 마지막 FC 레이어 제거 → feature extractor 형태로 사용
    resnet.fc = nn.Identity()
    return resnet


def replace_submodules(
    root_module: nn.Module,
    predicate: Callable[[nn.Module], bool],
    func: Callable[[nn.Module], nn.Module],
) -> nn.Module:
    """
    Replace all submodules in `root_module` that satisfy `predicate`
    with the result of `func(old_module)`.

    Args:
        root_module: 루트 모듈 (예: 전체 ResNet)
        predicate:   모듈을 받아 bool 반환. True인 모듈이 교체 대상.
        func:        (old_module) -> new_module 형태의 팩토리 함수.

    Returns:
        수정된 root_module (in-place로 바뀌지만, 편의상 반환도 해줌).
    """
    # 루트 자체를 바로 교체해야 하는 경우
    if predicate(root_module):
        return func(root_module)

    # 교체 대상 모듈의 "경로"들을 먼저 수집
    targets = [
        k.split(".")
        for k, m in root_module.named_modules(remove_duplicate=True)
        if predicate(m)
    ]

    # 실제 교체 작업 수행
    for *parent, k in targets:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule(".".join(parent))

        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)

        tgt_module = func(src_module)

        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)

    # 검증: 아직 predicate에 걸리는 모듈이 남아 있으면 안 됨
    remaining = [
        k
        for k, m in root_module.named_modules(remove_duplicate=True)
        if predicate(m)
    ]
    assert len(remaining) == 0, f"Some modules were not replaced: {remaining}"
    return root_module


def replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int = 16,
) -> nn.Module:
    """
    Replace all BatchNorm2d layers in `root_module` with GroupNorm.

    Args:
        root_module: 변환할 모델 (예: ResNet)
        features_per_group:
            GroupNorm에서 group 당 채널 수 기준.
            num_groups = num_features // features_per_group 로 설정.

    Returns:
        BatchNorm2d가 GroupNorm으로 모두 치환된 root_module
    """

    def predicate(m: nn.Module) -> bool:
        return isinstance(m, nn.BatchNorm2d)

    def factory(bn: nn.BatchNorm2d) -> nn.GroupNorm:
        return nn.GroupNorm(
            num_groups=bn.num_features // features_per_group,
            num_channels=bn.num_features,
        )

    replace_submodules(
        root_module=root_module,
        predicate=predicate,
        func=factory,
    )
    return root_module


__all__ = ["get_resnet", "replace_submodules", "replace_bn_with_gn"]
