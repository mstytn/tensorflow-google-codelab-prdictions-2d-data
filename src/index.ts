import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'

type Car = {
  Miles_per_Gallon: null | number,
  Horsepower: null | number
}

type CarParsed = {
  mpg: number,
  horsepower: number
}

type Data = CarParsed[]

interface TensorData {
  inputs: tf.Tensor<tf.Rank>
  labels: tf.Tensor<tf.Rank>
  inputMax: tf.Tensor<tf.Rank>
  inputMin: tf.Tensor<tf.Rank>
  labelMin: tf.Tensor<tf.Rank>
  labelMax: tf.Tensor<tf.Rank>
}

document.addEventListener('DOMContentLoaded', run);

async function getData() {
  const carsDataLink = 'https://storage.googleapis.com/tfjs-tutorials/carsData.json'
  const carsDataResponse = await fetch(carsDataLink);
  const carsData = await carsDataResponse.json();
  const cleaned = carsData.map((car: Car) => ({
    mpg: car.Miles_per_Gallon,
    horsepower: car.Horsepower,
  }))
  .filter((car: CarParsed) => (car.mpg != null && car.horsepower != null));

  return cleaned;
}
async function run() {
  // Load and plot the original input data that we are going to train on.
  const data = await getData();
  const values = data.map((d: CarParsed) => ({
    x: d.horsepower,
    y: d.mpg,
  }));

  tfvis.render.scatterplot(
    {name: 'Horsepower v MPG'},
    {values},
    {
      xLabel: 'Horsepower',
      yLabel: 'MPG',
      height: 300
    }
  );

  const model = createModel()
  tfvis.show.modelSummary({name: 'Model Summary'}, model)

  const tensorData: TensorData = convetToTensor(data)
  const { inputs, labels } = tensorData

  await trainModel(model, inputs, labels)
  console.log('Done Training')

  testModel(model, data, tensorData)
}



function createModel(): tf.Sequential {
  const model = tf.sequential()

  model.add(tf.layers.dense({inputShape: [1], units: 1, useBias: true}))
  model.add(tf.layers.dense({units: 50, activation: 'sigmoid'}));
  model.add(tf.layers.dense({units: 1, useBias: true}))

  return model
}

function convetToTensor<T extends Data>(data: T): TensorData {
  return tf.tidy(() => {
    
    tf.util.shuffle(data)
    const inputs = data.map(d => d.horsepower)
    const labels = data.map(d => d.mpg)

    const inputTensor = tf.tensor2d(inputs, [inputs.length, 1])
    const labelTensor = tf.tensor2d(labels, [labels.length, 1])

    const inputMax = inputTensor.max()
    const inputMin = inputTensor.min()
    const labelMax = labelTensor.max()
    const labelMin = labelTensor.min()

    const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin))
    const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin))

    return {
      inputs: normalizedInputs,
      labels: normalizedLabels,
      inputMax,
      inputMin,
      labelMax,
      labelMin
    }
  })
}



async function trainModel(model: tf.LayersModel, inputs: tf.Tensor<tf.Rank>, labels: tf.Tensor<tf.Rank>): Promise<tf.History> {
  model.compile({
    optimizer: tf.train.adam(),
    loss: tf.losses.meanSquaredError,
    metrics: ['mse']
  })

  const batchSize = 32
  const epochs = 50

  const modelFitArgs: tf.ModelFitArgs = {
    batchSize,
    epochs,
    shuffle: true,
    callbacks: tfvis.show.fitCallbacks(
      {name: 'Training Performance'},
      ['loss', 'mse'],
      { height: 200, callbacks: ['onEpochEnd'] }
    )
  }

  return await model.fit(inputs, labels, modelFitArgs)
}

function testModel(model: tf.LayersModel, inputData: Data, normalizationData: TensorData): void {
  const { inputMax, inputMin, labelMax, labelMin } = normalizationData
  const [xs, preds] = tf.tidy(() => {
    const xsNorm = tf.linspace(0, 1, 100)
    const predictions = model.predict(xsNorm.reshape([100, 1])) as tf.Tensor<tf.Rank>

    const unNormXs = xsNorm
      .mul(inputMax.sub(inputMin))
      .add(inputMin)
    
    const unNormPreds = predictions
      .mul(labelMax.sub(labelMin))
      .add(labelMin)

    return [unNormXs.dataSync(), unNormPreds.dataSync()]
      
  })

  const predictedPoints = Array.from(xs).map((val, i) => {
    return {x: val, y: preds[i]}
  })

  const originalPoints = inputData.map(d => ({
    x: d.horsepower, y: d.mpg
  }))

  tfvis.render.scatterplot(
    {name: 'Model Prdictions vs Original Data'},
    {values: [originalPoints, predictedPoints], series: ['original', 'predicted']},
    {
      xLabel: 'Horsepower',
      yLabel: 'MPG',
      height: 300
    }
  )
}