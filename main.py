#!/usr/bin/env python3
import click
from pathlib import Path
from src import (
    My_LicensePlate_Model,
    evaluate_model,
    log,
    plot_evaluation_results,
    process_video,
)
from src.train import train


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """License Plate Detection System using YOLO."""
    pass


@cli.command()
@click.option("--data", "-d", default="data/data.yaml", type=click.Path(), help="Path to data.yaml")
@click.option("--model", "-m", default="yolov8n.pt", help="Pretrained model name or path")
@click.option("--epochs", "-e", default=30, type=int, help="Number of epochs")
@click.option("--imgsz", "-s", default=640, type=int, help="Image size")
@click.option("--batch", "-b", default=8, type=int, help="Batch size")
@click.option("--device", default=0, help="Device (0 for GPU, 'cpu' for CPU)")
@click.option("--project", default="runs/detect", help="Project directory")
@click.option("--name", "-n", default="train", help="Experiment name")
def train_cmd(data, model, epochs, imgsz, batch, device, project, name):
    train(
        data_yaml=data,
        model_name=model,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
    )


@cli.command()
@click.option("--input", "-i", required=True, type=click.Path(exists=True), help="Input video file path")
@click.option("--output", "-o", required=True, type=click.Path(), help="Output video file path")
@click.option("--model", "-m", type=click.Path(exists=True), help="Path to model weights (.pt)")
@click.option("--conf", "-c", default=0.5, type=float, help="Confidence threshold (0.0-1.0)")
@click.option("--show/--no-show", default=False, help="Show preview window during processing")
def video(input, output, model, conf, show):
    log.info(f"CLI: video mode - input={input}, output={output}")
    process_video(
        input_path=input,
        output_path=output,
        model_path=model,
        conf_threshold=conf,
        show_preview=show,
    )


@cli.command()
@click.option("--model", "-m", required=True, type=click.Path(exists=True), help="Path to model weights (.pt)")
@click.option("--images", "-i", required=True, type=click.Path(exists=True), help="Path to images directory")
@click.option("--labels", "-l", required=True, type=click.Path(exists=True), help="Path to labels directory (YOLO format)")
@click.option("--iou", default=0.5, type=float, help="IoU threshold for true positive")
@click.option("--conf", default=0.5, type=float, help="Confidence threshold")
@click.option("--plot/--no-plot", default=False, help="Generate and save evaluation plots")
@click.option("--output", "-o", default="evaluation_results.json", type=click.Path(), help="Output JSON file for results")
def evaluate(model, images, labels, iou, conf, plot, output):
    log.info(f"CLI: evaluate mode - model={model}")

    results = evaluate_model(
        model_path=model,
        images_dir=images,
        labels_dir=labels,
        iou_threshold=iou,
        conf_threshold=conf,
    )

    if results:
        import json
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        log.info(f"Results saved to {output}")

        if plot:
            plot_path = output_path.with_suffix(".png")
            plot_evaluation_results(results, save_path=str(plot_path))

        click.echo("\n" + "=" * 50)
        click.echo("EVALUATION RESULTS")
        click.echo("=" * 50)
        click.echo(f"Total images: {results['total_images']}")
        click.echo(f"Precision:   {results['precision']:.4f}")
        click.echo(f"Recall:      {results['recall']:.4f}")
        click.echo(f"F1 Score:    {results['f1_score']:.4f}")
        click.echo(f"True Positives:  {results['true_positives']}")
        click.echo(f"False Positives: {results['false_positives']}")
        click.echo(f"False Negatives: {results['false_negatives']}")
        click.echo("=" * 50)


@cli.command()
@click.option("--model", "-m", type=click.Path(exists=True), help="Path to model weights (.pt)")
def info(model):
    if model:
        detector = My_LicensePlate_Model(model)
    else:
        detector = My_LicensePlate_Model()

    info = detector.get_model_info()
    click.echo(f"Model path: {info['model_path']}")
    click.echo(f"Model name: {info['model_name']}")
    click.echo(f"Classes: {info['classes']}")


if __name__ == "__main__":
    cli()