import torch
from avalanche.benchmarks.classic import CORe50
from avalanche.training.supervised import EWC
from avalanche.models import SimpleMLP
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, timing_metrics, forgetting_metrics
from avalanche.logging import InteractiveLogger

# Caricamento del dataset CORe50
benchmark = CORe50(scenario="ni")

# Definizione del modello
model = SimpleMLP(num_classes=benchmark.n_classes)

# Logger
interactive_logger = InteractiveLogger()

# Plugin per la valutazione
eval_plugin = EvaluationPlugin(
    accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    timing_metrics(epoch=True, epoch_running=True),
    forgetting_metrics(experience=True, stream=True),
    loggers=[interactive_logger]
)

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

# Addestramento ed evaluazione
for experience in benchmark.train_stream:
    print("Start training on experience ", experience.current_experience)
    cl_strategy.train(experience)
    print("End training on experience ", experience.current_experience)
    print("Computing accuracy on the whole test set")
    cl_strategy.eval(benchmark.test_stream)
