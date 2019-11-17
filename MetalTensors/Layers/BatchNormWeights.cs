using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Foundation;
using Metal;
using MetalPerformanceShaders;

using static MetalTensors.MetalExtensions;

namespace MetalTensors.Layers
{
    class BatchNormWeights : MPSCnnBatchNormalizationDataSource
    {
        readonly IMTLDevice device;

        readonly string label;
        private readonly int channels;

        MPSNNOptimizerAdam? updater;

        //readonly NSArray<MPSVector> momentumVectors;
        //readonly NSArray<MPSVector> velocityVectors;

        readonly OptimizableVector betaVector;
        readonly OptimizableVector gammaVector;
        readonly MPSCnnNormalizationGammaAndBetaState gammaAndBeta;
        readonly OptimizableVector meanVector;
        readonly OptimizableVector varianceVector;
        readonly MPSCnnNormalizationMeanAndVarianceState meanAndVariance;

        public override string Label => label;

        public override IntPtr Beta => betaVector.ValuePointer;
        public override IntPtr Gamma => betaVector.ValuePointer;

        public override IntPtr Mean => betaVector.ValuePointer;
        public override IntPtr Variance => betaVector.ValuePointer;

        public override nuint NumberOfFeatureChannels => (nuint)channels;

        public BatchNormWeights (int channels, string label, IMTLDevice device)
        {
            // https://github.com/apple/turicreate/blob/d332b2a856b0eadb97f6475a5728a624afe27e02/src/ml/neural_net/mps_weight.mm#L449

            this.device = device;
            this.channels = channels;

            if (channels <= 0)
                throw new ArgumentOutOfRangeException (nameof (channels), "Number of batch normalization channels must be > 0");

            this.label = string.IsNullOrEmpty (label) ? Guid.NewGuid ().ToString () : label;

            var lenWeights = channels;

            var vDescWeights = VectorDescriptor (lenWeights);

            betaVector = new OptimizableVector (device, vDescWeights, 0.0f);
            gammaVector = new OptimizableVector (device, vDescWeights, 1.0f);
            meanVector = new OptimizableVector (device, vDescWeights, 0.0f);
            varianceVector = new OptimizableVector (device, vDescWeights, 1.0f);

            SetVectorsModified ();

            gammaAndBeta = new MPSCnnNormalizationGammaAndBetaState (gammaVector.Value.Data, betaVector.Value.Data);
            meanAndVariance = new MPSCnnNormalizationMeanAndVarianceState (meanVector.Value.Data, varianceVector.Value.Data);

            //momentumVectors = biasVectors != null ?
            //    NSArray<MPSVector>.FromNSObjects (weightVectors.Momentum, biasVectors.Momentum) :
            //    NSArray<MPSVector>.FromNSObjects (weightVectors.Momentum);
            //velocityVectors = biasVectors != null ?
            //    NSArray<MPSVector>.FromNSObjects (weightVectors.Velocity, biasVectors.Velocity) :
            //    NSArray<MPSVector>.FromNSObjects (weightVectors.Velocity);

            SetOptimizationOptions (true, learningRate: Model.DefaultLearningRate);
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
                //Console.WriteLine ($"Load Conv2dDataSource {this.Label}");
                return true;
            }
        }

        public override void Purge ()
        {
            //Console.WriteLine ($"Purge Conv2dDataSource {this.Label}");
        }

        public override MPSCnnNormalizationGammaAndBetaState UpdateGammaAndBeta (IMTLCommandBuffer commandBuffer, MPSCnnBatchNormalizationState batchNormalizationState)
        {
            var u = updater;

            if (u != null) {
                //u.Encode (commandBuffer, batchNormalizationState, momentumVectors, velocityVectors, gammaAndBeta);
            }

            return gammaAndBeta;
        }

        public override MPSCnnNormalizationMeanAndVarianceState UpdateMeanAndVariance (IMTLCommandBuffer commandBuffer, MPSCnnBatchNormalizationState batchNormalizationState)
        {
            var u = updater;

            if (u != null) {
                //u.Encode (commandBuffer, batchNormalizationState, momentumVectors, velocityVectors, meanAndVariance);
            }

            return meanAndVariance;
        }

        public Dictionary<string, float[]> GetWeights ()
        {
            var r = new Dictionary<string, float[]> ();
            return r;
        }

        public bool WeightsAreValid ()
        {
            return betaVector.WeightsAreValid () &&
                gammaVector.WeightsAreValid () &&
                meanVector.WeightsAreValid () &&
                varianceVector.WeightsAreValid ();
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
