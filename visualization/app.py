import logging
import os
import threading
import time
from functools import partial
from pathlib import Path
from typing import Dict, List

import pygraphviz
import tensorflow as tf
from bokeh.document import Document
from bokeh.embed import server_document
from bokeh.layouts import row
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from bokeh.server.server import Server
from bokeh.themes import Theme
from flask import Flask, render_template, url_for
from flask_bootstrap import Bootstrap
from flask_socketio import SocketIO, disconnect, send
from pydantic import BaseSettings
from tornado import gen
from tornado.ioloop import IOLoop
from visualization.model import EpochPlot, Metrics, ProfilingPlots, TimeMetric, TrainTestPlots, load_metrics

app = Flask(__name__)
bootstrap = Bootstrap(app)
socketio = SocketIO(app)

METRICS: Dict[str, Metrics] = {}

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Settings(BaseSettings):
    log_path = Path('logdir')


@gen.coroutine
def update_aggregated_metrics(plots: TrainTestPlots, metric: Metrics) -> None:
    new_data_score = {'train_epoch': [metric.train_score.epoch], 'train_value': [metric.train_score.value]}
    new_data_loss = {'train_epoch': [metric.train_loss.epoch], 'train_value': [metric.train_loss.value]}
    if metric.validation_score is not None and metric.validation_loss is not None:
        new_data_score.update(
            {'validation_epoch': [metric.validation_score.epoch], 'validation_value': [metric.validation_score.value]}
        )
        new_data_loss.update(
            {'validation_epoch': [metric.validation_loss.epoch], 'validation_value': [metric.validation_loss.value]}
        )
    else:
        new_data_score.update({'validation_epoch': [], 'validation_value': []})
        new_data_loss.update({'validation_epoch': [], 'validation_value': []})

    plots.score.source.stream(new_data_score)
    plots.loss.source.stream(new_data_loss)


def load_model_graph(logdir: Path) -> pygraphviz.AGraph:
    while True:
        try:
            path = logdir / 'model.hd5'
            model = tf.keras.models.load_model(path)
            dot_graph = tf.keras.utils.model_to_dot(model, show_shapes=True)
            dot_graph = str(dot_graph).replace('?', 'data_size')
            graph = pygraphviz.AGraph()
            graph.from_string(dot_graph)
            return graph
        except OSError:
            time.sleep(1)


def load_data_for_model(doc: Document, plots: TrainTestPlots, logdir: Path) -> None:
    last_epoch = 0
    while doc.session_context:
        try:
            path = logdir / 'metrics' / f'epoch_{last_epoch}'
            new_metrics = load_metrics(str(path))
            last_epoch += 1
            doc.add_next_tick_callback(partial(update_aggregated_metrics, plots, new_metrics))
        except OSError:
            time.sleep(1)
    logger.info("Connection closed")


def load_profiling_data(doc: Document, plots: ProfilingPlots, logdir: Path, model_name: str) -> None:
    last_epoch = 0
    while doc.session_context:
        try:
            path = logdir / 'metrics' / f'epoch_{last_epoch}'
            new_metrics = load_metrics(str(path))
            last_epoch += 1
            doc.add_next_tick_callback(partial(update_profiling_metrics, plots, new_metrics))
            METRICS[model_name] = new_metrics
        except OSError:
            time.sleep(1)
    logger.info("Connection closed")


def get_plots() -> TrainTestPlots:
    plots = TrainTestPlots(
        score=EpochPlot(
            plot=figure(plot_height=150, plot_width=200, y_axis_label='Score', title="Score of the model"),
            source=ColumnDataSource(
                data={"train_epoch": [], "validation_epoch": [], "train_value": [], "validation_value": []}
            ),
        ),
        loss=EpochPlot(
            plot=figure(plot_height=150, plot_width=200, y_axis_label='Loss', title="Loss of the model"),
            source=ColumnDataSource(
                data={"train_epoch": [], "validation_epoch": [], "train_value": [], "validation_value": []}
            ),
        ),
    )
    plots.loss.plot.line(
        "train_epoch", "train_value", color="firebrick", source=plots.loss.source, legend_label='train'
    )
    plots.loss.plot.line(
        "validation_epoch", "validation_value", color="navy", source=plots.loss.source, legend_label='validation'
    )
    plots.score.plot.line(
        "train_epoch", "train_value", color="firebrick", source=plots.score.source, legend_label='train'
    )
    plots.score.plot.line(
        "validation_epoch", "validation_value", color="navy", source=plots.score.source, legend_label='validation'
    )

    for plot in [plots.loss.plot, plots.score.plot]:
        plot.legend.location = "bottom_left"

    return plots


def metrics_doc(doc: Document) -> None:
    settings = Settings()
    logdir = settings.log_path / doc.session_context.request.arguments['model'][0].decode('utf-8')
    plots = get_plots()
    doc.add_root(row(plots.loss.plot, plots.score.plot))
    doc.theme = Theme(filename="visualization/theme.yaml")
    threading.Thread(target=load_data_for_model, args=(doc, plots, logdir)).start()


def get_profiler_plots() -> ProfilingPlots:
    plots = ProfilingPlots(
        full_epoch=EpochPlot(
            plot=figure(title='Total duration for full epoch (ms)'),
            source=ColumnDataSource(data={'epoch': [], 'value': []}),
        )
    )

    plots.full_epoch.plot.line('epoch', 'value', color="navy", source=plots.full_epoch.source)

    return plots


@gen.coroutine
def update_profiling_metrics(plots: ProfilingPlots, metrics: Metrics) -> None:
    plots.full_epoch.source.stream({'epoch': [metrics.fit_metrics.epoch], 'value': [metrics.fit_metrics.last_duration]})


def profiler_doc(doc: Document) -> None:
    settings = Settings()
    model_name = doc.session_context.request.arguments['model'][0].decode('utf-8')
    logdir = settings.log_path / model_name
    plots = get_profiler_plots()
    doc.add_root(row(plots.full_epoch.plot))
    doc.theme = Theme(filename="visualization/theme.yaml")
    threading.Thread(target=load_profiling_data, args=(doc, plots, logdir, model_name)).start()


def get_models(log_path: Path) -> List[str]:
    if not log_path.exists():
        return []
    return list(map(lambda x: x.name, log_path.iterdir()))


@app.route('/', methods=['GET'])
def main_page() -> str:
    settings = Settings()
    model_names = get_models(settings.log_path)
    models = [{'name': model_name, 'href': url_for('model_page', name=model_name)} for model_name in model_names]
    return render_template("main.html", models=models)


@app.route('/model/<name>', methods=['GET'])
def model_page(name: str) -> str:
    script = server_document('http://localhost:5006/bkapp', arguments={'model': name})
    profiler_script = server_document('http://localhost:5006/profiler', arguments={'model': name})
    return render_template("embed.html", script=script, profiler_script=profiler_script, model=name)


@socketio.on('message')
def load_graph(msg: str) -> None:
    settings = Settings()
    model_name = msg
    graph = load_model_graph(settings.log_path / model_name)
    send(str(graph), callback=disconnect)


def format_metrics_map(m: Dict[str, TimeMetric]) -> str:
    return '\n'.join(
        [
            f'{str(value.average_duration).rjust(16)}ms: {name}'
            for name, value in sorted(m.items(), key=lambda x: x[1].average_duration, reverse=True)
        ]
    )


def format_metrics(metrics: Metrics) -> str:
    full = metrics.fit_metrics
    return f'''Metrics report:
    Full epoch          : average duration={full.average_duration}ms, last_duration={full.last_duration}ms
    Full batch          : average duration={metrics.fit_batch_metrics.average_duration}ms
    Gradient calculation: average duration={metrics.gradients_metrics.average_duration}ms
    Forward pass
{format_metrics_map(metrics.forward_pass_metrics)}
    Backward pass
{format_metrics_map(metrics.forward_pass_metrics)}
    Gradient step
{format_metrics_map(metrics.forward_pass_metrics)}
    Apply gradients
{format_metrics_map(metrics.forward_pass_metrics)}
'''


@socketio.on('message', namespace='/profiler_metrics')
def profiling_info(model_name: str) -> None:
    while True:
        metrics = METRICS.get(model_name)
        if metrics is not None:
            data = format_metrics(metrics)
            send(data)
        time.sleep(1)


def bk_worker() -> None:
    server = Server(
        {'/bkapp': metrics_doc, '/profiler': profiler_doc}, io_loop=IOLoop(), allow_websocket_origin=["127.0.0.1:8000"]
    )
    server.start()
    server.io_loop.start()


def run() -> None:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    threading.Thread(target=bk_worker).start()
    socketio.run(app, host='0.0.0.0', port=8000)
