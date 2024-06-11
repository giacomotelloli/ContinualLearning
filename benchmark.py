import torch
import torch.nn as nn
import torch.nn.functional as F
from avalanche.benchmarks import SplitMNIST
from avalanche.training.supervised import EWC
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, timing_metrics, forgetting_metrics
from avalanche.benchmarks.classic import CORe50
from avalanche.logging import InteractiveLogger


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=50):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Definizione del modello RNN
class SimpleRNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(28, 128, batch_first=True)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.squeeze(1)  # Rimuovere la dimensione del canale
        x, _ = self.rnn(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x

# Definizione del modello Vision Transformer (ViT)
class SimpleViT(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleViT, self).__init__()
        self.vit = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=True)
        self.vit.head = nn.Linear(self.vit.head.in_features, num_classes)

    def forward(self, x):
        x = F.interpolate(x, size=(224, 224))
        x = self.vit(x)
        return x

# Caricamento del dataset CORe50
benchmark = CORe50(scenario="ni")

# Logger
interactive_logger = InteractiveLogger()

# Definizione del plugin di valutazione
# Plugin per la valutazione
eval_plugin = EvaluationPlugin(
    accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    timing_metrics(epoch=True, epoch_running=True),
    forgetting_metrics(experience=True, stream=True),
    loggers=[interactive_logger]
)


# Lista dei modelli
models = [SimpleCNN(num_classes=benchmark.n_classes)]


# Iterazione sui modelli e allenamento
for model in models:
    
    print(f"Training model: {model.__class__.__name__}")
    # Definizione della strategia EWC
    cl_strategy = EWC(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9),
        criterion=torch.nn.CrossEntropyLoss(),
        ewc_lambda=0.4,  # lambda per regolarizzazione EWC
        train_mb_size=32,
        train_epochs=1,
        eval_mb_size=100,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        evaluator=eval_plugin
    )

    for experience in benchmark.train_stream:
        cl_strategy.train(experience)
        cl_strategy.eval(benchmark.test_stream)
