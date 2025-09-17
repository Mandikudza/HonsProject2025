import matplotlib.pyplot as plt
import json

def plot_training_history_matplotlib(history, model_name, output_path):
    """Generate and save a Matplotlib plot of training history."""
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_losses'], label='Train Loss')
    #plt.plot(history['test_losses'], label='Test Loss')
    plt.title(f'{model_name} Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()
    print(f"Training history plot saved to: {output_path}")

def generate_training_history_chartjs(history, model_name, output_path):
    """Generate a Chart.js JSON configuration for training history visualization."""
    chart_config = {
        "type": "line",
        "data": {
            "labels": [f"Epoch {i+1}" for i in range(len(history['train_losses']))],
            "datasets": [
                {
                    "label": "Train Loss",
                    "data": history['train_losses'],
                    "borderColor": "#1f77b4",
                    "backgroundColor": "rgba(31, 119, 180, 0.2)",
                    "fill": True
                },
                """{
                    "label": "Test Loss",
                    "data": history['test_losses'],
                    "borderColor": "#ff7f0e",
                    "backgroundColor": "rgba(255, 127, 14, 0.2)",
                    "fill": True
                }"""
            ]
        },
        "options": {
            "responsive": True,
            "title": {
                "display": True,
                "text": f"{model_name} Training History"
            },
            "scales": {
                "x": {
                    "title": {
                        "display": True,
                        "text": "Epoch"
                    }
                },
                "y": {
                    "title": {
                        "display": True,
                        "text": "Loss"
                    },
                    "beginAtZero": False
                }
            }
        }
    }
    with open(output_path, 'w') as f:
        json.dump(chart_config, f, indent=2)
    print(f"Training history Chart.js config saved to: {output_path}")
