import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import Function
import torch.nn.functional as F


def SqrtmLayer(var, iterN):
    return Sqrtm.apply(var, iterN)


class Sqrtm(Function):
    '''
    refer to Eq. (2-4)
    OR
        Peihua Li, Jiangtao Xie, Qilong Wang, Zilin Gao. "Towards Faster Training of Global Covariance Pooling Networks by Iterative
        Matrix Square Root Normalization" CVPR 2018
    '''
    @staticmethod
    def forward(ctx, input, iterN):
        x = input
        batchSize = x.data.shape[0]
        dim = x.data.shape[1]
        dtype = x.dtype
        I3 = 3.0 * torch.eye(dim, dim, device=x.device).view(1, dim, dim).repeat(batchSize, 1, 1).type(dtype)
        normA = (1.0 / 3.0) * x.mul(I3).sum(dim=1).sum(dim=1)
        A = x.div(normA.view(batchSize, 1, 1).expand_as(x))
        Y = torch.zeros(batchSize, iterN, dim, dim, requires_grad=False, device=x.device)
        Z = torch.eye(dim, dim, device=x.device).view(1, dim, dim).repeat(batchSize, iterN, 1, 1)
        if iterN < 2:
            ZY = 0.5 * (I3 - A)
            Y[:, 0, :, :] = A.bmm(ZY)
        else:
            ZY = 0.5 * (I3 - A)
            Y[:, 0, :, :] = A.bmm(ZY)
            Z[:, 0, :, :] = ZY
            for i in range(1, iterN - 1):
                ZY = 0.5 * (I3 - Z[:, i - 1, :, :].bmm(Y[:, i - 1, :, :]))
                Y[:, i, :, :] = Y[:, i - 1, :, :].bmm(ZY)
                Z[:, i, :, :] = ZY.bmm(Z[:, i - 1, :, :])
            ZY = 0.5 * Y[:, iterN - 2, :, :].bmm(I3 - Z[:, iterN - 2, :, :].bmm(Y[:, iterN - 2, :, :]))
        y = ZY * torch.sqrt(normA).view(batchSize, 1, 1).expand_as(x)
        ctx.save_for_backward(input, A, ZY, normA, Y, Z)
        ctx.iterN = iterN
        return y

    @staticmethod
    def backward(ctx, grad_output):
        input, A, ZY, normA, Y, Z = ctx.saved_tensors
        iterN = ctx.iterN
        x = input
        batchSize = x.data.shape[0]
        dim = x.data.shape[1]
        dtype = x.dtype
        der_postCom = grad_output * torch.sqrt(normA).view(batchSize, 1, 1).expand_as(x)
        der_postComAux = (grad_output * ZY).sum(dim=1).sum(dim=1).div(2 * torch.sqrt(normA))
        I3 = 3.0 * torch.eye(dim, dim, device=x.device).view(1, dim, dim).repeat(batchSize, 1, 1).type(dtype)
        if iterN < 2:
            der_NSiter = 0.5 * (der_postCom.bmm(I3 - A) - A.bmm(der_postCom))
        else:
            dldY = 0.5 * (der_postCom.bmm(I3 - Y[:, iterN - 2, :, :].bmm(Z[:, iterN - 2, :, :])) -
                          Z[:, iterN - 2, :, :].bmm(Y[:, iterN - 2, :, :]).bmm(der_postCom))
            dldZ = -0.5 * Y[:, iterN - 2, :, :].bmm(der_postCom).bmm(Y[:, iterN - 2, :, :])
            for i in range(iterN - 3, -1, -1):
                YZ = I3 - Y[:, i, :, :].bmm(Z[:, i, :, :])
                ZY = Z[:, i, :, :].bmm(Y[:, i, :, :])
                dldY_ = 0.5 * (dldY.bmm(YZ) -
                               Z[:, i, :, :].bmm(dldZ).bmm(Z[:, i, :, :]) -
                               ZY.bmm(dldY))
                dldZ_ = 0.5 * (YZ.bmm(dldZ) -
                               Y[:, i, :, :].bmm(dldY).bmm(Y[:, i, :, :]) -
                               dldZ.bmm(ZY))
                dldY = dldY_
                dldZ = dldZ_
            der_NSiter = 0.5 * (dldY.bmm(I3 - A) - dldZ - A.bmm(dldY))
        grad_input = der_NSiter.div(normA.view(batchSize, 1, 1).expand_as(x))
        grad_aux = der_NSiter.mul(x).sum(dim=1).sum(dim=1)
        for i in range(batchSize):
            grad_input[i, :, :] += (der_postComAux[i] \
                                    - grad_aux[i] / (normA[i] * normA[i])) \
                                   * torch.ones(dim, device=x.device).diag()
        return grad_input, None


class Sqrtm_autograd(nn.Module):
    '''
    refer to Eq. (2-4)
    the same as Sqrtm implemented by autograd package of PyTorch
    the difference between Sqrtm and Sqrtm_autograd has been checked in check_sqrtm()
    '''
    def __init__(self, norm_type, num_iter):
        super(Sqrtm_autograd, self).__init__()
        self.norm_type = norm_type
        self.num_iter = num_iter

    def forward(self, A):
        dtype = A.dtype
        batchSize = A.data.shape[0]
        dim = A.data.shape[1]
        normA = []
        traces = []
        # pre normalization
        if self.norm_type == 'AF':
            normA = A.mul(A).sum(dim=1).sum(dim=1).sqrt()
            Y = A.div(normA.view(batchSize, 1, 1).expand_as(A))
        elif self.norm_type == 'AT':
            diags = []
            for i in range(batchSize):
                diags.append(torch.unsqueeze(torch.diag(A[i, :, :]), dim=0))
            diags = torch.cat(diags)
            traces = torch.unsqueeze(torch.sum(diags, dim=-1, keepdim=True), dim=-1)  # nx1x1
            Y = A.div(traces.expand_as(A))
        else:
            raise NameError('invalid normalize type {}'.format(self.norm_type))

        # Iteration
        I = Variable(torch.eye(dim, dim).view(1, dim, dim).
                     repeat(batchSize, 1, 1).type(dtype), requires_grad=False)
        Z = Variable(torch.eye(dim, dim).view(1, dim, dim).
                     repeat(batchSize, 1, 1).type(dtype), requires_grad=False)

        for i in range(self.num_iter):
            T = 0.5 * (3.0 * I - Z.bmm(Y))
            Y = Y.bmm(T)
            Z = T.bmm(Z)

        # post normalization
        if self.norm_type == 'AF':
            sA = Y * torch.sqrt(normA).view(batchSize, 1, 1).expand_as(A)
        else:
            sA = Y * torch.sqrt(traces).expand_as(A)
        del I, Z
        return sA


class RingLoss(nn.Module):
    def __init__(self, type='L2', loss_weight=1.0):
        super(RingLoss, self).__init__()
        self.radius = nn.Parameter(torch.Tensor(1))
        self.radius.data.fill_(-1)
        self.loss_weight = loss_weight
        self.type = type

    def forward(self, x):
        x = x.pow(2).sum(dim=1).pow(0.5)
        if self.radius.data[0] < 0: # Initialize the radius with the mean feature norm of first iteration
            self.radius.data.fill_(x.mean().data[0])
        if self.type == 'L1': # Smooth L1 Loss
            loss1 = F.smooth_l1_loss(x, self.radius.expand_as(x)).mul_(self.loss_weight)
            loss2 = F.smooth_l1_loss(self.radius.expand_as(x), x).mul_(self.loss_weight)
            ringloss = loss1 + loss2
        elif self.type == 'auto': # Divide the L2 Loss by the feature's own norm
            diff = x.sub(self.radius.expand_as(x)) / (x.mean().detach().clamp(min=0.5))
            diff_sq = torch.pow(torch.abs(diff), 2).mean()
            ringloss = diff_sq.mul_(self.loss_weight)
        else: # L2 Loss, if not specified
            diff = x.sub(self.radius.expand_as(x))
            diff_sq = torch.pow(torch.abs(diff), 2).mean()
            ringloss = diff_sq.mul_(self.loss_weight)
        return ringloss


def check_sqrtm():
    a = torch.randn((2, 256, 256))
    a_auto = Variable(torch.matmul(a, torch.transpose(a, 1, 2)), requires_grad=True)
    a_back = Variable(torch.matmul(a, torch.transpose(a, 1, 2)), requires_grad=True)
    sqrt_auto = Sqrtm_autograd(norm_type='AT', num_iter=5)

    r_auto = sqrt_auto(a_auto)
    r_back = SqrtmLayer(a_back, 5)

    diff = r_auto - r_back
    diff = torch.sum(diff.mul(diff))
    print('forward difference {}'.format(diff))

    loss_auto = torch.sum(r_auto)
    loss_back = torch.sum(r_back)

    loss_auto.backward()
    loss_back.backward()

    diff_grad = a_auto.grad - a_back.grad
    diff_grad = torch.sum(diff_grad.mul(diff_grad))
    print('backward difference {}'.format(diff_grad))
    print('finished')


if __name__ == '__main__':
    check_sqrtm()

