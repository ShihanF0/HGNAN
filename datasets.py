import torch_geometric as pyg
from get_hypergraph import *

import os
import torch

def get_data(data_name, processed_data_dir='processed_data', model_name='HGNAN-node', train_size=0.5, val_size=0.25, batch_size=64, seed=None, s_values=[1]):
    
    print(f'Loading {data_name} dataset')
    train_loader, val_loader, test_loader = None, None, None
    num_features = None
    
    s_max = max(s_values) if s_values else 0
    processed_file_path = f'{processed_data_dir}/{data_name}_smax{s_max}.pt'

    if not os.path.exists(processed_file_path):
        if model_name == 'HGNAN-node':
          if data_name in ['zoo', 'Mushroom', 'NTU2012']:
              data = load_LE_dataset(data_name=data_name, train_size=train_size, val_size=val_size, s_values=s_values, processed_file_path=processed_file_path)
          else:
              data = load_heter_dataset(data_name=data_name, s_values=s_values, processed_file_path=processed_file_path, compute_distances=True)
        else:
            data = get_hypergraph(dataset=data_name, train_size=train_size, val_size=val_size, s_values=s_values, processed_file_path=processed_file_path)
    else:
        data = torch.load(processed_file_path, weights_only=False) # <-- Use the new path variable
        print(f'Loaded {data_name} dataset from cache: {processed_file_path}')
    print("Preprocess Finished!")

    if model_name == 'HGNAN-node':
      if data_name == 'Mushroom':
          num_classes = 2
      elif data_name == 'NTU2012':
          num_classes = 67
      elif data_name == 'zoo':
          num_classes = 7
      elif data_name == 'actor':
          num_classes = 3
      elif data_name == 'pokec':
          num_classes = 2
    else:
        num_classes = 2

    num_nodes = data.x.shape[0]
    labels = data.y

    indices = np.arange(num_nodes)
    train_val_size = train_size + val_size
    test_size = 1 - train_val_size
    val_ratio = val_size / train_val_size

    train_val_idx, test_idx, train_val_y, test_y = train_test_split(
        indices, labels, test_size=test_size, stratify=labels, random_state=seed
    )

    train_idx, val_idx, train_y, val_y = train_test_split(
        train_val_idx, train_val_y, test_size=val_ratio, stratify=train_val_y, random_state=seed
    )

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    
    if train_loader is None:
        train_data = data
        val_data = data
        test_data = data
        train_loader = pyg.loader.DataLoader([train_data], batch_size=batch_size)
        val_loader = pyg.loader.DataLoader([val_data], batch_size=batch_size)
        test_loader = pyg.loader.DataLoader([test_data], batch_size=batch_size)

    if num_features is None:
        num_features = train_data.x.shape[1]
    print(torch.unique(data.dist_mats['s1']))
    return train_loader, val_loader, test_loader, num_features, num_classes

