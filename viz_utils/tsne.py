import torch
import numpy as np
import os


from sklearn.manifold import TSNE


def extract_features(model, loader):

    labels = []

    def forward_hook():
        def _hook(module, inputs, outputs):
            print("Called Once ", module)
            print(inputs[0].size(), outputs.size())
            if not hasattr(module, 'features'):
                module.features = inputs[0].detach().clone().cpu().numpy().ravel().reshape(inputs[0].size()[0], -1)
            else:
                module.features = np.concatenate((module.features, inputs[0].detach().clone().cpu().numpy().ravel().reshape(inputs[0].size()[0], -1)), axis=0)
        return _hook
    
    model.eval()

    #This will attach the hook to the last linear layer. Its input will be used as feature
    linear_layer = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            linear_layer = module

    handle = linear_layer.register_forward_hook(forward_hook())

    device = next(model.parameters())[0].device
    with torch.no_grad():
        for ind, (x,y) in enumerate(loader):
            x, y = x.to(device), y.to(device)

            _ = model(x)
            print(linear_layer.features.shape)
            labels.extend(list(y.detach().clone().cpu().numpy().ravel()))

    features = linear_layer.features

    handle.remove()
    del(linear_layer.features)

    return np.array(features), np.array(labels)

def get_tsne(model, loader, n_components=2, perplexity=3):

    X, Y = extract_features(model, loader)
    X = TSNE(n_components=n_components, learning_rate='auto', init='random', perplexity=perplexity).fit_transform(X)

    return X, Y

