# import copy
# import time
# from copy import deepcopy

# import torch
# import torch.nn as nn
# import numpy as np

# import utils
# from .impl import iterative_unlearn


# def discretize(x):
#     return torch.round(x * 255) / 255


# def _get_final_linear(model: nn.Module) -> nn.Linear:
#     """
#     Robustly fetch the final Linear layer (classifier).
#     Prefers `model.fc` if present; otherwise grabs the last nn.Linear found.
#     """
#     if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
#         return model.fc
#     last_linear = None
#     # reverse traversal to get the last Linear quickly
#     for m in reversed(list(model.modules())):
#         if isinstance(m, nn.Linear):
#             last_linear = m
#             break
#     if last_linear is None:
#         raise RuntimeError("No final nn.Linear layer found in the model.")
#     return last_linear


# @torch.no_grad()
# def distance_to_decision_boundaries(x: torch.Tensor,
#                                     y: torch.Tensor,
#                                     model: nn.Module) -> torch.Tensor:
#     """
#     Vectorized computation of the nearest class (by boundary distance) for each sample.
#     Returns a LongTensor of shape (B,) with the argmin j != y for each row.
#     """
#     device = x.device
#     model.eval()

#     # Get logits
#     # inference_mode is slightly faster / safer than no_grad for pure inference
#     with torch.inference_mode():
#         z = model(x)  # (B, C)

#     # Use provided y as the 'reference' class per your original code
#     # (you set pred_classes = y to move away from the forget class)
#     c = y  # (B,)

#     # Extract classifier weights
#     final_linear = _get_final_linear(model)
#     W = final_linear.weight  # (C, D)
#     # Precompute ||w_c - w_j|| as a (C, C) matrix
#     # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b
#     WWt = W @ W.t()                    # (C, C)
#     w_sq = WWt.diag().unsqueeze(0)     # (1, C)
#     # distances squared matrix:
#     dist2 = (w_sq + w_sq.t()) - 2.0 * WWt
#     # numerical safety: clamp to >= 0
#     dist2 = dist2.clamp_min(0.0)
#     Wdiff_norm = torch.sqrt(dist2 + 1e-12)  # (C, C)

#     # Numerator: |z_c - z_j| for all j
#     # z_c: (B, 1)
#     z_c = z.gather(dim=1, index=c.view(-1, 1))
#     num = (z_c - z).abs()  # (B, C)

#     # Denominator per sample: row Wdiff_norm[c[i], :]
#     # Build via index_select
#     denom = Wdiff_norm.index_select(0, c)  # (B, C)
#     # Avoid divide-by-zero (shouldn’t happen often, but just in case)
#     denom = denom + 1e-12

#     D = num / denom  # (B, C)
#     # Mask out self-class j=c with +inf so argmin excludes it
#     D.scatter_(1, c.view(-1, 1), float('inf'))

#     # Nearest class by boundary distance
#     nearest = torch.argmin(D, dim=1)  # (B,)

#     return nearest.to(device)


# @iterative_unlearn
# def boundary_shrink_iter(
#     data_loaders, model, criterion, optimizer, epoch, args, mask=None, test_model=None
# ):
#     assert test_model is not None  # kept for API compatibility, even though unused now

#     # ---- Use the forget loader directly; no need to deepcopy dataset or rebuild loader each time
#     forget_loader = data_loaders["forget"]

#     # If you truly need a separate loader (e.g., different shuffle/batch size), build once outside
#     train_loader = torch.utils.data.DataLoader(
#         forget_loader.dataset,
#         batch_size=getattr(args, "batch_size", 128),
#         shuffle=True,
#         num_workers=getattr(args, "num_workers", 4),
#         pin_memory=True,
#         drop_last=False,
#     )

#     losses = utils.AverageMeter()
#     top1 = utils.AverageMeter()

#     model.train()

#     use_amp = torch.cuda.is_available() and getattr(args, "use_amp", True)
#     scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

#     start = time.time()

#     for i, (image, target) in enumerate(train_loader):
#         if epoch < getattr(args, "warmup", 0):
#             utils.warmup_lr(epoch, i + 1, optimizer, one_epoch_step=len(train_loader), args=args)

#         image = image.to(next(model.parameters()).device, non_blocking=True)
#         target = target.to(image.device, non_blocking=True)

#         optimizer.zero_grad(set_to_none=True)

#         with torch.cuda.amp.autocast(enabled=use_amp):
#             output_clean = model(image)  # (B, C)
#             # Vectorized target based on nearest boundary class
#             output_target = distance_to_decision_boundaries(image, target, model)
#             loss = criterion(output_clean, output_target)

#         # Backward with AMP
#         scaler.scale(loss).backward()

#         # Optional gradient masking
#         if mask:
#             with torch.no_grad():
#                 for name, param in model.named_parameters():
#                     if param.grad is not None:
#                         param.grad.mul_(mask[name])

#         scaler.step(optimizer)
#         scaler.update()

#         # Metrics (detach to avoid dtype/AMP issues)
#         losses.update(float(loss.detach()), image.size(0))
#         prec1 = utils.accuracy(output_clean.detach(), target)[0]
#         top1.update(float(prec1), image.size(0))

#         if (i + 1) % getattr(args, "print_freq", 50) == 0:
#             end = time.time()
#             print(
#                 "Epoch: [{0}][{1}/{2}]\t"
#                 "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
#                 "Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
#                 "Time {3:.2f}".format(
#                     epoch, i, len(train_loader), end - start, loss=losses, top1=top1
#                 )
#             )
#             start = time.time()

#     print("train_accuracy {top1.avg:.3f}".format(top1=top1))
#     return top1.avg


# def boundary_dist(data_loaders, model: nn.Module, criterion, args, mask=None):
#     # `test_model` isn’t used anymore; we keep the API to avoid breaking callers.
#     # If you want to drop it fully, also update `@iterative_unlearn` and callers.
#     device = next(model.parameters()).device
#     test_model = copy.deepcopy(model).to(device)
#     return boundary_shrink_iter(
#         data_loaders, model, criterion, args, mask, test_model=test_model
#     )


############################ Loss- margin based ################################

import copy
import time
from copy import deepcopy

import torch
import torch.nn as nn
import numpy as np

import utils
from .impl import iterative_unlearn


def discretize(x):
    return torch.round(x * 255) / 255


def _get_final_linear(model: nn.Module) -> nn.Linear:
    """
    Robustly fetch the final Linear layer (classifier).
    Prefers `model.fc` if present; otherwise grabs the last nn.Linear found.
    """
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        return model.fc
    last_linear = None
    # reverse traversal to get the last Linear quickly
    for m in reversed(list(model.modules())):
        if isinstance(m, nn.Linear):
            last_linear = m
            break
    if last_linear is None:
        raise RuntimeError("No final nn.Linear layer found in the model.")
    return last_linear


@torch.no_grad()
def distance_to_decision_boundaries(x: torch.Tensor,
                                    y: torch.Tensor,
                                    model: nn.Module) -> torch.Tensor:
    """
    Vectorized computation of the nearest class (by boundary distance) for each sample.
    Returns a LongTensor of shape (B,) with the argmin j != y for each row.
    (Kept for compatibility; not used by the new loss.)
    """
    device = x.device
    model.eval()

    with torch.inference_mode():
        z = model(x)  # (B, C)

    c = y  # (B,)

    final_linear = _get_final_linear(model)
    W = final_linear.weight  # (C, D)

    WWt = W @ W.t()                    # (C, C)
    w_sq = WWt.diag().unsqueeze(0)     # (1, C)
    dist2 = (w_sq + w_sq.t()) - 2.0 * WWt
    dist2 = dist2.clamp_min(0.0)
    Wdiff_norm = torch.sqrt(dist2 + 1e-12)  # (C, C)

    z_c = z.gather(dim=1, index=c.view(-1, 1))
    num = (z_c - z).abs()  # (B, C)

    denom = Wdiff_norm.index_select(0, c)  # (B, C)
    denom = denom + 1e-12

    D = num / denom  # (B, C)
    D.scatter_(1, c.view(-1, 1), float('inf'))

    nearest = torch.argmin(D, dim=1)  # (B,)

    return nearest.to(device)


def margin_to_boundary_loss(logits: torch.Tensor,
                            y: torch.Tensor,
                            model: nn.Module,
                            eps: float = 1e-12) -> torch.Tensor:
    """
    Normalized margin loss that pulls each sample toward all decision boundaries
    that touch its current (forget) class y:

        L_margin = (1/(C-1)) * sum_{j != y} ((z_y - z_j) / ||w_y - w_j||)^2

    Where z are logits and w_* are rows of the final linear layer (classifier head).
    """
    # logits: [B, C], y: [B]
    B, C = logits.shape

    # Gather final-layer weights; fall back to unnormalized if not found
    W = None
    try:
        final_linear = _get_final_linear(model)
        W = final_linear.weight  # [C, D]
    except Exception:
        W = None

    if W is not None:
        # Precompute ||w_a - w_b|| as a (C, C) matrix
        WWt = W @ W.t()                          # [C, C]
        w_sq = WWt.diag().unsqueeze(0)           # [1, C]
        dist2 = (w_sq + w_sq.t()) - 2.0 * WWt
        dist2 = dist2.clamp_min(0.0)
        Wdiff_norm = torch.sqrt(dist2 + eps)     # [C, C]
        denom = Wdiff_norm.index_select(0, y)    # [B, C] rows for each y_i
    else:
        # No linear head found; use unnormalized margins (degenerates to squared margin)
        denom = torch.ones_like(logits)

    # z_y - z_j for all j
    z_y = logits.gather(1, y.view(-1, 1))        # [B, 1]
    margins = z_y - logits                       # [B, C]
    # zero-out self term so it won't contribute
    margins.scatter_(1, y.view(-1, 1), 0.0)

    # normalized squared margins
    nm = margins / (denom + eps)                 # [B, C]
    loss = (nm.pow(2).sum(dim=1) / (C - 1)).mean()
    return loss


@iterative_unlearn
def boundary_shrink_iter(
    data_loaders, model, criterion, optimizer, epoch, args, mask=None, test_model=None
):
    """
    Updated training loop:
      - Uses ONLY the normalized margin-to-boundary loss on the forget set.
      - Leaves everything else unchanged (AMP, metrics, logging).
      - `criterion` is ignored for the main loss (kept in signature for compatibility).
    """
    assert test_model is not None  # kept for API compatibility

    forget_loader = data_loaders["forget"]

    train_loader = torch.utils.data.DataLoader(
        forget_loader.dataset,
        batch_size=getattr(args, "batch_size", 128),
        shuffle=True,
        num_workers=getattr(args, "num_workers", 4),
        pin_memory=True,
        drop_last=False,
    )

    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()

    model.train()

    use_amp = torch.cuda.is_available() and getattr(args, "use_amp", True)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start = time.time()

    for i, (image, target) in enumerate(train_loader):
        if epoch < getattr(args, "warmup", 0):
            utils.warmup_lr(epoch, i + 1, optimizer, one_epoch_step=len(train_loader), args=args)

        image = image.to(next(model.parameters()).device, non_blocking=True)
        target = target.to(image.device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(image)  # (B, C)
            # --- NEW: normalized margin-to-boundary loss ---
            loss = margin_to_boundary_loss(logits, target, model)

        scaler.scale(loss).backward()

        # Optional gradient masking
        if mask:
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad.mul_(mask[name])

        scaler.step(optimizer)
        scaler.update()

        # Metrics (for monitoring only): original accuracy against true labels
        losses.update(float(loss.detach()), image.size(0))
        prec1 = utils.accuracy(logits.detach(), target)[0]
        top1.update(float(prec1), image.size(0))

        if (i + 1) % getattr(args, "print_freq", 50) == 0:
            end = time.time()
            print(
                "Epoch: [{0}][{1}/{2}]\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
                "Time {3:.2f}".format(
                    epoch, i, len(train_loader), end - start, loss=losses, top1=top1
                )
            )
            start = time.time()

    print("train_accuracy {top1.avg:.3f}".format(top1=top1))
    return top1.avg


def boundary_dist(data_loaders, model: nn.Module, criterion, args, mask=None):
    # `test_model` isn’t used anymore; we keep the API to avoid breaking callers.
    device = next(model.parameters()).device
    test_model = copy.deepcopy(model).to(device)
    return boundary_shrink_iter(
        data_loaders, model, criterion, args, mask, test_model=test_model
    )
