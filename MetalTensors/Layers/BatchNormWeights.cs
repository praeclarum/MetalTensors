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
        readonly BatchNormWeights batchNormWeights;
        readonly IMTLDevice device;

        MPSNNOptimizerAdam? updater;

        //readonly NSArray<MPSVector> momentumVectors;
        //readonly NSArray<MPSVector> velocityVectors;

        readonly OptimizableVector betaVector;
        readonly OptimizableVector gammaVector;
        readonly MPSCnnNormalizationGammaAndBetaState gammaAndBeta;
        readonly OptimizableVector meanVector;
        readonly OptimizableVector varianceVector;
        readonly MPSCnnNormalizationMeanAndVarianceState meanAndVariance;        

        public override string Label => batchNormWeights.Label;

        public override IntPtr Beta => betaVector.ValuePointer;
        public override IntPtr Gamma => gammaVector.ValuePointer;

        public override IntPtr Mean => meanVector.ValuePointer;
        public override IntPtr Variance => varianceVector.ValuePointer;

        public override nuint NumberOfFeatureChannels => (nuint)batchNormWeights.FeatureChannels;

        public BatchNormDataSource (BatchNormWeights batchNormWeights, IMTLDevice device)
        {
            // https://github.com/apple/turicreate/blob/d332b2a856b0eadb97f6475a5728a624afe27e02/src/ml/neural_net/mps_weight.mm#L449

            this.batchNormWeights = batchNormWeights;
            this.device = device;

            var lenWeights = batchNormWeights.FeatureChannels;

            var vDescWeights = VectorDescriptor (lenWeights);

            betaVector = new OptimizableVector (device, vDescWeights, batchNormWeights.Beta);
            gammaVector = new OptimizableVector (device, vDescWeights, batchNormWeights.Gamma);
            meanVector = new OptimizableVector (device, vDescWeights, batchNormWeights.MovingMean);
            varianceVector = new OptimizableVector (device, vDescWeights, batchNormWeights.MovingVariance);

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
