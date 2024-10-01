# import torch
# from torch import nn
# import tqdm
# from torch.utils.data import DataLoader, SubsetRandomSampler, TensorDataset
#
# import mlflow
# import mlflow.pytorch
# from mlflow.models import infer_signature


# class BasicNet(nn.Module):
#     def __init__(self):
#         super(BasicNet, self).__init__()

#         layer = 512
#         self.layers = nn.Sequential(
#             nn.Linear(280, layer),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=0.1),
#             nn.BatchNorm1d(layer),
#             nn.Linear(layer, layer),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=0.1),
#             nn.BatchNorm1d(layer),
#             nn.Linear(layer, layer),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=0.1),
#             nn.BatchNorm1d(layer),
#             nn.Linear(layer, 3),
#             nn.ReLU(inplace=True),
#         )

#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 torch.nn.init.kaiming_normal_(
#                     m.weight, mode="fan_out", nonlinearity="relu"
#                 )
#                 torch.nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm1d):
#                 torch.nn.init.constant_(m.weight, 1)
#                 torch.nn.init.constant_(m.bias, 0)

#     def forward(self, x):
#         x = self.layers(x)
#         return x


# def train(
#     model,
#     optimizer,
#     loss_fn,
#     train_loader,
#     val_loader,
#     writer=None,
#     epochs=1,
#     device="cpu",
#     model_path="best.pt",
#     scheduler=None,
#     tolerance=-1,
#     tolerance_delta=1e-4,
# ):
#     best = 0.0

#     not_improving = 0
#     last_loss = None

#     # iterating over epochs
#     for epoch in range(epochs):
#         # training loop description
#         train_loop = tqdm(
#             enumerate(train_loader, 0), total=len(train_loader), desc=f"Epoch {epoch}"
#         )
#         model.train()
#         train_loss = 0.0

#         # iterate over dataset
#         for data in train_loop:
#             # Write your code here
#             # Move data to a device, do forward pass and loss calculation, do backward pass and run optimizer
#             data = data[1]
#             inputs, labels = data[0].to(device), data[1].to(device)

#             optimizer.zero_grad()

#             output = model(inputs)

#             loss = loss_fn(output, labels)
#             train_loss += loss.item()

#             loss.backward()
#             optimizer.step()

#             train_loop.set_postfix({"loss": loss.item()})

#             if scheduler:
#                 scheduler.step()

#         mlflow.log_metric("loss", f"{train_loss / len(train_loader):6f}", step=epoch)

#         # Validation
#         correct = 0
#         total = 0

#         with torch.no_grad():
#             model.eval()  # evaluation mode
#             val_loop = tqdm(enumerate(val_loader, 0), total=len(val_loader), desc="Val")
#             for data in val_loop:
#                 data = data[1]
#                 inputs, labels = data[0].to(device), data[1].to(device)
#                 # Write your code here
#                 output = model(inputs)
#                 loss = loss_fn(output, labels).item()

#                 # Get predictions and compare them with labels
#                 pred = output.argmax(dim=1, keepdim=True)
#                 correct += pred.eq(labels.view_as(pred)).sum().item()
#                 total += len(labels)

#                 val_loop.set_postfix({"acc": correct / total})

#             val_acc = correct / total
#             mlflow.log_metric("validation loss", f"{loss / len(val_loader):6f}", step=epoch)
#             mlflow.log_metric("validation accuracy", f"{val_acc:2f}", step=epoch)
#             if val_acc > best:
#                 torch.save(model.parameters, "model_best.pt")
#                 torch.save(optimizer.state_dict(), "opimizer.pt")
#                 best = correct / total
#         if epoch != 0:
#             if abs(train_loss - last_loss) < tolerance_delta:
#                 not_improving += 1
#                 if not_improving == tolerance:
#                     print("Stop due to early reaching tolerance_delta")
#                     break
#             else:
#                 not_improving = 0
#         last_loss = train_loss

#     print(best)


# def overfit_single_batch(
#     train_loader,
#     model,
#     loss_fn,
#     device="cpu",
#     epochs=1,
#     batch_size=1,
#     lr=1e-2,
# ):
#     data_iter = iter(train_loader)
#     single_batch = next(data_iter)
#     inputs, labels = single_batch[0].to(device), single_batch[1].to(device)

#     single_batch_loader = DataLoader(
#         list(zip(inputs, labels)), batch_size=batch_size, shuffle=True
#     )

#     model.to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     model.train()

#     for epoch in range(epochs):
#         train_loop = tqdm(
#             enumerate(single_batch_loader, 0),
#             total=len(single_batch_loader),
#             desc=f"Epoch {epoch}",
#         )
#         train_loss = 0.0
#         for i, (batch_inputs, batch_labels) in train_loop:
#             optimizer.zero_grad()

#             output = model(inputs)
#             loss = loss_fn(output, labels)

#             train_loss += loss.item()

#             loss.backward()
#             optimizer.step()

#             train_loop.set_postfix({"loss": loss.item()})

#         ls = train_loss / len(single_batch_loader)
#         print(f"Epoch {epoch} Training Loss: {ls}")
#         mlflow.log_metric("Overfitting Loss", f"{ls:2f}", step=epoch)


# def mlflow_training():
#     device = "mps"

#     epochs = 1

#     tolerance = 7
#     tolerance_delta = 1e-4

#     model_name = "basic_model"
#     model_alias = "latest"

#     client = mlflow.MlflowClient()
#     mlflow.set_tracking_uri(uri="http://127.0.0.1:8090")

#     overfit_single_batch(
#         model=BasicNet(),
#         loss_fn=nn.CrossEntropyLoss(),
#         train_loader=train_loader,
#         device="mps",
#         epochs=5,
#         batch_size=1,
#         lr=1e-2,
#     )
#     client = mlflow.MlflowClient()
#     with mlflow.start_run() as run:
#         train(
#             model=model,
#             optimizer=optimizer,
#             loss_fn=loss_function,
#             train_loader=train_loader,
#             val_loader=val_loader,
#             device=device,
#             epochs=epochs,
#             tolerance=tolerance,
#             tolerance_delta=tolerance_delta,
#         )
#         data_iter = iter(train_loader)
#         inputs, labels = next(data_iter)
#         signature = infer_signature(
#             inputs.numpy(), model(inputs.to(device)).detach().cpu().numpy()
#         )
#         model_info = mlflow.pytorch.log_model(
#             pytorch_model=model,
#             artifact_path="models",
#             signature=signature,
#             input_example=inputs.numpy(),
#             registered_model_name=model_name
#         )
#         client.set_registered_model_alias(model_name, model_alias,  model_info.registered_model_version)
#         model_ver = client.get_model_version_by_alias(model_name, model_alias)
#         torch_model = mlflow.pytorch.load_model(f"models:/{model_name}@{model_alias}")
#         torch_model.to(device)

#         val_iter = iter(val_loader)
#         X_val, y_val = next(val_iter)

#         X_val = X_val.to(device)
#         y_val = y_val.to(device)

#         raw_predictions = torch_model(X_val)
#         pred = raw_predictions.argmax(dim=1, keepdim=True)
#         predictions = pred.eq(y_val.view_as(pred)).sum().item()

#         print(predictions)

#         eval_data = pd.DataFrame(X_val.cpu())
#         eval_data["label"] = y_val.cpu()
#         eval_data["predictions"] = predictions
#         print(eval_data.shape)

#         results = mlflow.evaluate(
#             data=eval_data,
#             model_type="classifier",
#             targets="label",
#             predictions="predictions",
#             evaluators=["default"],
#         )

#     print(f"metrics:\n{results.metrics}")
#     print(f"artifacts:\n{results.artifacts}")
