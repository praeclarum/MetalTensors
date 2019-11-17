using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Foundation;
using Metal;
using MetalPerformanceShaders;

using static MetalTensors.MetalExtensions;

namespace MetalTensors.Layers
{
    public class BatchNormWeights
    {
        public Tensor Beta { get; }
        public Tensor Gamma { get; }
        public Tensor MovingMean { get; }
        public Tensor MovingVariance { get; }
        public string Label { get; }
        public int FeatureChannels { get; }

        readonly ConcurrentDictionary<IntPtr, BatchNormDataSource> deviceWeights =
            new ConcurrentDictionary<IntPtr, BatchNormDataSource> ();

        public BatchNormWeights (string label, int channels)
        {
            if (channels <= 0)
                throw new ArgumentOutOfRangeException (nameof (channels), "Number of batch normalization channels must be > 0");

            Label = label;
            FeatureChannels = channels;
            Beta = Tensor.Zeros (FeatureChannels);
            Gamma = Tensor.Ones (FeatureChannels);
            MovingMean = Tensor.Zeros (FeatureChannels);
            MovingVariance = Tensor.Ones (FeatureChannels);
        }

        public MPSCnnBatchNormalizationDataSource GetDataSource (IMTLDevice device)
        {
            var key = device.Handle;
            if (deviceWeights.TryGetValue (key, out var w))
                return w;

            w = new BatchNormDataSource (this, device);

            if (deviceWeights.TryAdd (key, w))
                return w;
            return deviceWeights[key];
        }
    }

    class BatchNormDataSource : MPSCnnBatchNormalizationDataSource
    {
        const float bnRunningUpdateMomentum = 0.9f;

        readonly BatchNormWeights batchNormWeights;
        readonly IMTLDevice device;

        readonly MPSVectorDescriptor vectorDescriptor;

        readonly OptimizableVector betaVector;
        readonly OptimizableVector gammaVector;
        readonly MPSCnnNormalizationGammaAndBetaState gammaAndBeta;

        readonly MPSVector meanVector;
        readonly MPSVector varianceVector;
        readonly MPSCnnNormalizationMeanAndVarianceState meanAndVariance;

        MPSNNOptimizerAdam? updater;
        readonly MPSNNOptimizerStochasticGradientDescent meanUpdater;
        readonly MPSNNOptimizerStochasticGradientDescent varianceUpdater;

        readonly NSArray<MPSVector> momentumVectors;
        readonly NSArray<MPSVector> velocityVectors;

        public override string Label => batchNormWeights.Label;

        public override IntPtr Beta => betaVector.ValuePointer;
        public override IntPtr Gamma => gammaVector.ValuePointer;

        public override IntPtr Mean => meanVector.Data.Contents;
        public override IntPtr Variance => varianceVector.Data.Contents;

        public override nuint NumberOfFeatureChannels => (nuint)batchNormWeights.FeatureChannels;

        public BatchNormDataSource (BatchNormWeights batchNormWeights, IMTLDevice device)
        {
            // https://github.com/apple/turicreate/blob/d332b2a856b0eadb97f6475a5728a624afe27e02/src/ml/neural_net/mps_weight.mm#L449

            this.batchNormWeights = batchNormWeights;
            this.device = device;

            vectorDescriptor = VectorDescriptor (batchNormWeights.FeatureChannels);

            betaVector = new OptimizableVector (device, vectorDescriptor, batchNormWeights.Beta);
            gammaVector = new OptimizableVector (device, vectorDescriptor, batchNormWeights.Gamma);
            meanVector = Vector (device, vectorDescriptor, batchNormWeights.MovingMean);
            varianceVector = Vector (device, vectorDescriptor, batchNormWeights.MovingVariance);

            SetVectorsModified ();

            gammaAndBeta = new MPSCnnNormalizationGammaAndBetaState (gammaVector.Value.Data, betaVector.Value.Data);
            meanAndVariance = new MPSCnnNormalizationMeanAndVarianceState (meanVector.Data, varianceVector.Data);

            momentumVectors = NSArray<MPSVector>.FromNSObjects (gammaVector.Momentum, betaVector.Momentum);
            velocityVectors = NSArray<MPSVector>.FromNSObjects (gammaVector.Velocity, betaVector.Velocity);

            SetOptimizationOptions (true, learningRate: Model.DefaultLearningRate);

            /*
              A note on how the batch norm update works.
              What we want is to perform:
                  value(t+1) = mu * value(t) + (1 - mu) * statistic(t)
              Value is the batch norm statistics (global mean or variance), mu is the
              momentum of the update, and statistic is either the mean or variance of the
              current batch.
              We use an SGD optimizer without moment and L2 weight decay, which performs:
                  value(t+1) = value(t) - learningRate * (gradient(t) + value(t) * regularizationScale)
              Solving this gives:
                  learningRate = -(1 - mu)
                  regularizationScale = -1
                  gradient(t) = statistic(t)
            */
            var bnRunningOptDesc = new MPSNNOptimizerDescriptor (
                learningRate: -(1 - bnRunningUpdateMomentum),
                gradientRescale: 1.0f,
                regularizationType: MPSNNRegularizationType.L2,
                regularizationScale: -1.0f);
            meanUpdater = new MPSNNOptimizerStochasticGradientDescent (
                device: device,
                momentumScale: 0.0f,
                useNestrovMomentum: false,
                optimizerDescriptor: bnRunningOptDesc);
            varianceUpdater = new MPSNNOptimizerStochasticGradientDescent (
                device: device,
                momentumScale: 0.0f,
                useNestrovMomentum: false,
                optimizerDescriptor: bnRunningOptDesc);
        }

        public void SetOptimizationOptions (bool trainable, float learningRate)
        {
            if (trainable) {
                var odesc = new MPSNNOptimizerDescriptor (learningRate, 1.0f, MPSNNRegularizationType.None, 1.0f);
                updater = new MPSNNOptimizerAdam (
                    device,
                    beta1: 0.9f, beta2: 0.999f, epsilon: 1e-8f,
                    timeStep: 0,
                    optimizerDescriptor: odesc);
            }
            else {
                updater = null;
            }
        }

        [DebuggerHidden]
        public override bool Load {
            get {
                //Console.WriteLine ($"Load BatchNormDataSource {this.Label}");
                return true;
            }
        }

        public override void Purge ()
        {
            //Console.WriteLine ($"Purge BatchNormDataSource {this.Label}");
        }

        public override MPSCnnNormalizationGammaAndBetaState UpdateGammaAndBeta (IMTLCommandBuffer commandBuffer, MPSCnnBatchNormalizationState batchNormalizationState)
        {
            var u = updater;

            if (u != null) {
                //
                // Update mean and variance
                //
                using var meanState = new MPSVector (batchNormalizationState.Mean, vectorDescriptor);
                using var varianceState = new MPSVector (batchNormalizationState.Variance, vectorDescriptor);
                meanUpdater.Encode (commandBuffer, inputGradientVector: meanState, inputValuesVector: meanVector, null, meanVector);
                varianceUpdater.Encode (commandBuffer, inputGradientVector: varianceState, inputValuesVector: varianceVector, null, varianceVector);

                //
                // Update gamma and beta
                // This update must come last, since the MPS API we use will decrement read counts.
                //
                u.Encode (commandBuffer, batchNormalizationState, momentumVectors, velocityVectors, gammaAndBeta);
            }

            return gammaAndBeta;
        }

        //public Dictionary<string, float[]> GetWeights ()
        //{
        //    var r = new Dictionary<string, float[]> ();
        //    return r;
        //}

        public bool WeightsAreValid ()
        {
            return betaVector.IsValid () &&
                gammaVector.IsValid () &&
                meanVector.IsValid () &&
                varianceVector.IsValid ();
        }

        void SetVectorsModified ()
        {
            betaVector.DidModify ();
            gammaVector.DidModify ();
            meanVector.DidModify ();
            varianceVector.DidModify ();
        }

#if PB_SERIALIZATION
        public NetworkData.DataSource GetData (bool includeTrainingParameters)
        {
            var c = new NetworkData.ConvolutionDataSource {
                Weights = weightVectors.GetData (includeTrainingParameters: includeTrainingParameters),
                Biases = biasVectors.GetData (includeTrainingParameters: includeTrainingParameters),
            };
            return new NetworkData.DataSource {
                Convolution = c
            };
        }

        public void SetData (NetworkData.DataSource dataSource)
        {
            var c = dataSource.Convolution;
            if (c == null)
                return;

            weightVectors.SetData (c.Weights);
            biasVectors.SetData (c.Biases);

            SetVectorsModified ();
        }
#endif
    }
}
