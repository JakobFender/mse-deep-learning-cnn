import torch


class VisionTrainer:

    def __init__(self, model, criterion, optimizer, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def _train(self, train_loader):
        """
        Run one full epoch over the training set.

        Returns:
            avg_loss (float): mean loss across all batches
            accuracy (float): fraction of correctly classified samples
        """
        self.model.train()  # Enable training mode (dropout and batch-norm)

        running_loss = 0.0
        correct_preds = 0
        total_samples = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            # Move data to the same device as the model
            images, labels = images.to(self.device), labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()  # Clear gradients from the previous step
            logits = self.model(images)  # Get raw class scores
            loss = self.criterion(logits, labels)  # Compute loss

            # Backward pass
            loss.backward()  # Compute gradients via backprop
            self.optimizer.step()  # Update weights

            # Metrics accumulation
            running_loss += loss.item() * images.size(0)  # Weighted by batch size
            preds = logits.argmax(dim=1)  # Predicted class indices
            correct_preds += (preds == labels).sum().item()
            total_samples += labels.size(0)

        avg_loss = running_loss / total_samples
        accuracy = correct_preds / total_samples
        return avg_loss, accuracy

    def _evaluate(self, validation_loader):
        """
        Evaluate the model on any DataLoader (validation or test).

        Returns:
            avg_loss  (float): mean loss
            accuracy  (float): fraction correct
            all_preds (list):  predicted class indices for each sample
            all_labels(list):  true class indices for each sample
        """
        self.model.eval()  # Enable evaluation mode (Disable dropout)

        running_loss = 0.0
        correct_preds = 0
        total_samples = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():  # No gradients needed — saves memory and time
            for images, labels in validation_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                logits = self.model(images)
                loss = self.criterion(logits, labels)

                running_loss += loss.item() * images.size(0)
                preds = logits.argmax(dim=1)
                correct_preds += (preds == labels).sum().item()
                total_samples += labels.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = running_loss / total_samples
        accuracy = correct_preds / total_samples
        return avg_loss, accuracy, all_preds, all_labels

    def run(self, train_loader, validation_loader, num_epochs=1):

        # Lists to store per-epoch metrics for plotting later
        history = {
            "train_loss": [], "train_acc": [],
            "test_loss": [], "test_acc": [],
        }

        print("\n" + "=" * 55)
        print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | {'Test Loss':>9} | {'Test Acc':>8}")
        print("=" * 55)

        for epoch in range(1, num_epochs + 1):
            # One full pass over the training data
            train_loss, train_acc = self._train(train_loader)

            # Evaluate on the test set at the end of every epoch
            test_loss, test_acc, _, _ = self._evaluate(validation_loader)

            # Record metrics
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["test_loss"].append(test_loss)
            history["test_acc"].append(test_acc)

            print(f"{epoch:>6} | {train_loss:>10.4f} | {train_acc:>9.4f} | {test_loss:>9.4f} | {test_acc:>8.4f}")
