// wake-word-trainer.js — Rathor™ browser-native custom wake-word training (TensorFlow.js)
// MIT license — Eternal Thriving Grandmasterism

import * as tf from '@tensorflow/tfjs';
import * as speechCommands from '@tensorflow-models/speech-commands';

export class WakeWordTrainer {
  constructor(orchestrator) {
    this.orchestrator = orchestrator;
    this.recognizer = null;
    this.model = null;
    this.customWord = 'rathor'; // default — can be changed by user
    this.samples = {
      positive: [],
      negative: []
    };
    this.isTraining = false;
    this.minSamples = 20;
  }

  async init() {
    try {
      // Load base speech commands model (transfer learning base)
      this.recognizer = speechCommands.create('BROWSER_FFT');
      await this.recognizer.ensureModelLoaded();
      console.log("Base speech commands model loaded — ready for transfer learning.");
      return true;
    } catch (err) {
      console.error("Failed to load base model:", err);
      return false;
    }
  }

  // Record short audio clip (1–2 seconds) for training
  async recordSample(isPositive = true) {
    if (!this.recognizer) await this.init();

    try {
      const spectrogram = await this.recognizer.listen(
        async ({ spectrogram: { data, frameSize, numFrames } }) => {
          // Convert raw spectrogram to tensor
          const tensor = tf.tensor4d(data, [1, numFrames, frameSize, 1]);
          if (isPositive) {
            this.samples.positive.push(tensor);
          } else {
            this.samples.negative.push(tensor);
          }
          console.log(`${isPositive ? 'Positive' : 'Negative'} sample recorded — total: ${this.samples[isPositive ? 'positive' : 'negative'].length}`);
        },
        { probabilityThreshold: 0.0, invokeCallbackOnNoiseAndUnknown: true, overlapFactor: 0.5 }
      );

      // Stop listening after \~2 seconds
      setTimeout(() => spectrogram.stop(), 2000);
    } catch (err) {
      console.error("Recording failed:", err);
    }
  }

  // Train custom detector
  async train() {
    if (this.isTraining) return "Training already in progress...";

    if (this.samples.positive.length < this.minSamples || this.samples.negative.length < this.minSamples) {
      return `Need at least ${this.minSamples} positive and negative samples each. Current: ${this.samples.positive.length} pos / ${this.samples.negative.length} neg.`;
    }

    this.isTraining = true;

    try {
      // Create transfer learning model from base
      const baseModel = this.recognizer.model;
      const newModel = tf.sequential();

      // Freeze base layers
      baseModel.layers.forEach(layer => {
        layer.trainable = false;
        newModel.add(layer);
      });

      // Add new head for 2 classes: rathor vs background
      newModel.add(tf.layers.dense({ units: 2, activation: 'softmax' }));

      newModel.compile({
        optimizer: 'adam',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
      });

      // Prepare training data
      const xsPos = tf.concat(this.samples.positive).reshape([-1, ...this.samples.positive[0].shape.slice(1)]);
      const xsNeg = tf.concat(this.samples.negative).reshape([-1, ...this.samples.negative[0].shape.slice(1)]);
      const xs = tf.concat([xsPos, xsNeg]);

      const ysPos = tf.oneHot(tf.zeros([xsPos.shape[0]]), 2);
      const ysNeg = tf.oneHot(tf.ones([xsNeg.shape[0]]), 2);
      const ys = tf.concat([ysPos, ysNeg]);

      // Train
      await newModel.fit(xs, ys, {
        epochs: 20,
        batchSize: 8,
        callbacks: {
          onEpochEnd: (epoch, logs) => {
            console.log(`Epoch ${epoch}: loss = ${logs.loss.toFixed(4)}, acc = ${logs.acc.toFixed(4)}`);
          }
        }
      });

      this.model = newModel;
      this.isTraining = false;

      return `Custom wake-word model trained! Positive samples: ${this.samples.positive.length}, Negative: \( {this.samples.negative.length}. Ready to listen for " \){this.customWord}". ⚡️`;
    } catch (err) {
      this.isTraining = false;
      console.error("Training failed:", err);
      return "Training failed — check console for details. Mercy asks: try again with more samples?";
    }
  }

  // Inference — run continuously when active
  async listenForWakeWord() {
    if (!this.model) return "No trained model yet — record samples first.";

    while (true) {
      try {
        const { score } = await this.recognizer.recognize(this.model);
        if (score[0] > 0.85) { // threshold for "rathor"
          console.log("Wake-word detected!");
          this.orchestrator.orchestrate("wake detected");
          await this.orchestrator.voice.start(); // activate full immersion
        }
      } catch (err) {
        console.warn("Wake-word inference error:", err);
      }
      await new Promise(r => setTimeout(r, 500)); // 2 fps check
    }
  }
}

export default WakeWordTrainer;
