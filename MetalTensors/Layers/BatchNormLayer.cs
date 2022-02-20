using System;
using System.Collections.Concurrent;
using System.Diagnostics;
using Foundation;
using Metal;
using MetalPerformanceShaders;

using static MetalTensors.MetalHelpers;

namespace MetalTensors.Layers
{
    public class BatchNormLayer : WeightsLayer
    {
        // https://github.com/apple/turicreate/blob/98b61f551f26429e05025c79785d7b4b0ef50295/src/ml/neural_net/mps_layers.mm#L443
        public const float DefaultEpsilon = 0.001f;

        public int FeatureChannels { get; }
        public float Epsilon { get; }

        public override int MinInputCount => 1;

        public BatchNormLayer (int featureChannels, float epsilon = DefaultEpsilon, string? name = null, bool isTrainable = true)
            : base (name, isTrainable: isTrainable)
        {
            FeatureChannels = featureChannels;
            Epsilon = epsilon;
        }

        public override Config Config => base.Config.Update (new Config {
            { "featureChannels", FeatureChannels },
            { "epsilon", Epsilon },
        });

        public override int[] GetOutputShape (params Tensor[] inputs)
        {
            return inputs[0].Shape;
        }

        protected override MPSNNFilterNode CreateFilterNode ((MPSNNImageNode ImageNode, int[] Shape)[] inputs, IMTLDevice device)
        {
            return new MPSCnnBatchNormalizationNode (inputs[0].ImageNode, GetDataSource<BatchNormDataSource> (device));
        }

        protected override IWeightsDataSource CreateDataSource (IMTLDevice device)
        {
            return new BatchNormDataSource (this, device);
        }
    }

    class BatchNormDataSource : MPSCnnBatchNormalizationDataSource, IWeightsDataSource
    {
        const float bnRunningUpdateMomentum = 0.9f;

        readonly BatchNormLayer batchNormWeights;
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

        public override string Label => batchNormWeights.Name;

        public override IntPtr Beta => betaVector.ValuePointer;
        public override IntPtr Gamma => gammaVector.ValuePointer;

        public override IntPtr Mean => meanVector.Data.Contents;
        public override IntPtr Variance => varianceVector.Data.Contents;

        public override nuint NumberOfFeatureChannels => (nuint)batchNormWeights.FeatureChannels;

        public override float Epsilon => batchNormWeights.Epsilon;

        public BatchNormDataSource (BatchNormLayer batchNormWeights, IMTLDevice device)
        {
            // https://github.com/apple/turicreate/blob/d332b2a856b0eadb97f6475a5728a624afe27e02/src/ml/neural_net/mps_weight.mm#L449

            this.batchNormWeights = batchNormWeights;
            this.device = device;

            vectorDescriptor = VectorDescriptor (batchNormWeights.FeatureChannels);

            betaVector = new OptimizableVector (device, vectorDescriptor);
            gammaVector = new OptimizableVector (device, vectorDescriptor);
            meanVector = Vector (vectorDescriptor, device);
            varianceVector = Vector (vectorDescriptor, device);
            batchNormWeights.Weights.Read ("Beta", betaVector, initialValue: 0.0f);
            batchNormWeights.Weights.Read ("Gamma", gammaVector, initialValue: 1.0f);
            batchNormWeights.Weights.Read ("Mean", meanVector, initialValue: 0.0f);
            batchNormWeights.Weights.Read ("Variance", varianceVector, initialValue: 1.0f);

            gammaAndBeta = new MPSCnnNormalizationGammaAndBetaState (gammaVector.Value.Data, betaVector.Value.Data);
            meanAndVariance = new MPSCnnNormalizationMeanAndVarianceState (meanVector.Data, varianceVector.Data);

            momentumVectors = NSArray<MPSVector>.FromNSObjects (gammaVector.Momentum, betaVector.Momentum);
            velocityVectors = NSArray<MPSVector>.FromNSObjects (gammaVector.Velocity, betaVector.Velocity);

            SetOptimizationOptions (true, learningRate: Optimizer.DefaultLearningRate);

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

            if (u != null && batchNormalizationState.Mean is IMTLBuffer mean && batchNormalizationState.Variance is IMTLBuffer variance) {
                //
                // Update mean and variance
                //
                using var meanState = new MPSVector (mean, vectorDescriptor);
                using var varianceState = new MPSVector (variance, vectorDescriptor);
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

        public bool WeightsAreFinite ()
        {
            return betaVector.IsFinite () &&
                gammaVector.IsFinite () &&
                meanVector.IsFinite () &&
                varianceVector.IsFinite ();
        }

        void MarkAsModifiedByCpu ()
        {
            betaVector.MarkAsModifiedByCpu ();
            gammaVector.MarkAsModifiedByCpu ();
            meanVector.MarkAsModifiedByCpu ();
            varianceVector.MarkAsModifiedByCpu ();
        }
    }
}
