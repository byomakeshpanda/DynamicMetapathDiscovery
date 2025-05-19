import torch
import torch.nn as nn

def train_model(data, model, epochs=100, model_name="dynamic"):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    best_val = 0

    # Ensure CPU
    data = data.cpu()
    model = model.cpu()

    # Verify features
    for ntype in data.node_types:
        if not hasattr(data[ntype], 'x'):
            data[ntype].x = torch.ones(data[ntype].num_nodes, 1)

    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data)
        
        # Calculate loss
        loss = criterion(
            out[data['author'].train_mask],
            data['author'].y[data['author'].train_mask]
        )
        
        # Backprop
        loss.backward()
        optimizer.step()

        # Validation
        with torch.no_grad():
            pred = out.argmax(dim=1)
            val_acc = (pred[data['author'].val_mask] == data['author'].y[data['author'].val_mask]).float().mean()
            
            if val_acc > best_val:
                best_val = val_acc
                torch.save(model.state_dict(), f'{model_name}.pth')

        print(f'Epoch {epoch:03d} | Loss: {loss:.4f} | Val: {val_acc:.4f}')

    model.load_state_dict(torch.load(f'{model_name}.pth'))
    return model