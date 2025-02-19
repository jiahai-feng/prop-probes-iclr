import torch
import einops
from pathlib import Path

class das_form:
    def __init__(self, das_dir):
        name_subspace = torch.load(Path(das_dir) / "name_subspace.pt").cuda()
        attr_subspace = torch.load(Path(das_dir) / "attr_subspace.pt").cuda()
        self.form = name_subspace @ attr_subspace.T
        self.U = name_subspace
        self.S = torch.ones(name_subspace.shape[1], device=name_subspace.device)
        self.Vh = attr_subspace.T

class hessian_1_1:
    def __init__(self, hessian):
        name_width, name_dim, attr_width, attr_dim = hessian.shape
        assert name_width == 1 and attr_width == 1
        self.form = hessian[0, :, 0, :].cuda()
        self.U, self.S, self.Vh = torch.linalg.svd(self.form)


class hessian_2_1:
    def __init__(self, hessian):
        name_width, name_dim, attr_width, attr_dim = hessian.shape
        assert name_width == 2 and attr_width == 1
        hessian = hessian.cuda()
        self.form = hessian[0, :, 0, :] + hessian[1, :, 0, :]
        self.U, self.S, self.Vh = get_stacked_subspace(hessian)
        self.Vh = self.Vh[1]
        self.S = self.S[1]


class default:
    def __init__(self, hessian):
        name_dim, attr_dim = hessian.shape
        hessian = hessian.cuda()
        self.form = hessian
        self.U, self.S, self.Vh = torch.linalg.svd(self.form)


def torch_ridge(X, Y, alpha, driver="gels"):
    """
    X: [num samples, x features]
    Y: [num samples, y features]
    alpha: ridge weight i.e. ... + alpha ||w||^2

    returns [x features, y features] (note that scikit learn returns the opposite)
    """
    # c/f https://gist.github.com/myazdani/3d8a00cf7c9793e9fead1c89c1398f12
    lhs = X.T @ X + alpha * torch.eye(X.shape[1], device=X.device)
    rhs = X.T @ Y
    (solution, residuals, rank, singular_values) = torch.linalg.lstsq(lhs, rhs)
    return solution


def torch_rrr(X, Y, alpha, rank):
    """
    c/f https://github.com/krey/rrpy/blob/main/rrpy/reduced_rank_ridge.py
    c/f https://dept.stat.lsa.umich.edu/~jizhu/pubs/Mukherjee-SADM11.pdf

    returns [x features, y features]
    """
    beta_ridge = torch_ridge(X, Y, alpha)
    Lambda = torch.eye(X.shape[1], device=X.device) + alpha**0.5
    X_star = torch.concatenate((X, Lambda))
    Y_star = X_star @ beta_ridge
    _, _, Vt = torch.linalg.svd(Y_star, full_matrices=False)
    return (beta_ridge @ Vt.T[:, :rank]) @ Vt[:rank, :]


class freeze_attr:
    def __init__(self, hessian, alpha=1e3, max_rank=150):
        xes, yes, xpes, ypes = hessian
        ypes = ypes.cuda()[:, 0, :]
        xes = xes.cuda()[:, 0, :]
        xpes = xpes.cuda()[:, 0, :]
        yes = yes.cuda()[:, 0, :]
        self.form = torch_rrr(xes, ypes, alpha=alpha, rank=max_rank)
        self.U, self.S, self.Vh = torch.linalg.svd(self.form)


class freeze_name:
    def __init__(self, hessian, alpha=1e3, max_rank=150):
        xes, yes, xpes, ypes = hessian
        ypes = ypes.cuda()[:, 0, :]
        xes = xes.cuda()[:, 0, :]
        xpes = xpes.cuda()[:, 0, :]
        yes = yes.cuda()[:, 0, :]
        self.form = torch_rrr(yes, xpes, alpha=alpha, rank=max_rank).T
        self.U, self.S, self.Vh = torch.linalg.svd(self.form)

class random_form:
    def __init__(self, d_model):
        g = torch.Generator()
        g.manual_seed(42)
        Q, R = torch.linalg.qr(torch.randn(d_model, d_model, generator=g).cuda())
        self.form = torch.eye(d_model, device=Q.device)
        self.U, self.S, self.Vh = Q, torch.ones(d_model, device=Q.device), Q.T.clone()

def get_stacked_subspace(hessian):
    svds = [
        torch.linalg.svd(
            einops.rearrange(
                hessian[name_pos], "name_dim 1 attr_dim -> name_dim attr_dim"
            )
        )
        for name_pos in range(2)
    ]
    U2 = torch.stack([U for U, S, Vh in svds])
    S2 = torch.stack([S for U, S, Vh in svds])
    Vh2 = torch.stack([Vh for U, S, Vh in svds])
    return U2, S2, Vh2


def process_form(form, form_type):
    if form_type == "hessian_1_1":
        return hessian_1_1(form)
    elif form_type == "hessian_2_1":
        return hessian_2_1(form)
    elif form_type == "freeze_attr":
        return freeze_attr(form)
    elif form_type == "freeze_name":
        return freeze_name(form)
    elif form_type == "default":
        return default(form)
    else:
        raise Exception(f"Unknown form_type {form_type}")

def get_form(form_type, form_path=None, das_path=None):
    if form_type == "random":
        return random_form(5120)
    elif form_type == "das":
        return das_form(das_path)
    else:
        form = torch.load(form_path)
        return process_form(form, form_type)