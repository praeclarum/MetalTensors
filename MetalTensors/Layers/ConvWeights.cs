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
    public class ConvWeights
    {
        public string Label { get; }
        public int InChannels { get; }
        public int OutChannels { get; }
        public int SizeX { get; }
        public int SizeY { get; }
        public int StrideX { get; }
        public int StrideY { get; }
        public bool Bias { get; }
        public WeightsInit WeightsInit { get; }
        public float BiasInit { get; }

        readonly ConcurrentDictionary<IntPtr, ConvDataSource> deviceWeights =
            new ConcurrentDictionary<IntPtr, ConvDataSource> ();

        public ConvWeights (string label, int inChannels, int outChannels, int kernelSizeX, int kernelSizeY, int strideX, int strideY, bool bias, WeightsInit weightsInit, float biasInit)
        {
            if (inChannels <= 0)
                throw new ArgumentOutOfRangeException (nameof (inChannels), "Number of convolution input channels must be > 0");
            if (outChannels <= 0)
                throw new ArgumentOutOfRangeException (nameof (inChannels), "Number of convolution output channels must be > 0");

            Label = label;
            InChannels = inChannels;
            OutChannels = outChannels;
            SizeX = kernelSizeX;
            SizeY = kernelSizeY;
            StrideX = strideX;
            StrideY = strideY;
            Bias = bias;
            WeightsInit = weightsInit;
            BiasInit = biasInit;
            //Beta = Tensor.Zeros (FeatureChannels);
            //Gamma = Tensor.Ones (FeatureChannels);
            //MovingMean = Tensor.Zeros (FeatureChannels);
            //MovingVariance = Tensor.Ones (FeatureChannels);
        }

        public MPSCnnConvolutionDataSource GetDataSource (IMTLDevice device)
        {
            var key = device.Handle;
            if (deviceWeights.TryGetValue (key, out var w))
                return w;

            w = new ConvDataSource (this, device);

            if (deviceWeights.TryAdd (key, w))
                return w;
            return deviceWeights[key];
        }
    }

    class ConvDataSource : MPSCnnConvolutionDataSource
    {
        readonly IMTLDevice device;

        readonly string label;
        readonly bool bias;
        private readonly float biasInit;
        readonly MPSCnnConvolutionDescriptor descriptor;

        nuint updateCount;
        MPSNNOptimizerAdam? updater;

        readonly OptimizableVector weightVectors;
        readonly OptimizableVector? biasVectors;
        readonly MPSCnnConvolutionWeightsAndBiasesState convWtsAndBias;
        readonly NSArray<MPSVector> momentumVectors;
        readonly NSArray<MPSVector> velocityVectors;

        public override string Label => label;

        public override MPSCnnConvolutionDescriptor Descriptor => descriptor;

        public override MPSDataType DataType => MPSDataType.Float32;

        public override IntPtr Weights => weightVectors.ValuePointer;

        public override IntPtr BiasTerms => biasVectors != null ? biasVectors.ValuePointer : IntPtr.Zero;

        public ConvDataSource (ConvWeights convWeights, IMTLDevice device)
            : this (convWeights.InChannels, convWeights.OutChannels,
                    convWeights.SizeX, convWeights.SizeY, convWeights.StrideX, convWeights.StrideY,
                    convWeights.Bias, convWeights.WeightsInit, convWeights.BiasInit,
                    convWeights.Label, device)
        {
        }

        public ConvDataSource (int inChannels, int outChannels, int kernelSizeX, int kernelSizeY, int strideX, int strideY, bool bias, WeightsInit weightsInit, float biasInit, string label, IMTLDevice device)
        {
            this.device = device;

            if (inChannels <= 0)
                throw new ArgumentOutOfRangeException (nameof (inChannels), "Number of convolution input channels must be > 0");
            if (outChannels <= 0)
                throw new ArgumentOutOfRangeException (nameof (inChannels), "Number of convolution output channels must be > 0");

            descriptor = MPSCnnConvolutionDescriptor.CreateCnnConvolutionDescriptor (
                (System.nuint)kernelSizeX, (System.nuint)kernelSizeY,
                (System.nuint)inChannels,
                (System.nuint)outChannels);
            descriptor.StrideInPixelsX = (nuint)strideX;
            descriptor.StrideInPixelsY = (nuint)strideY;
            this.bias = bias;
            this.biasInit = biasInit;
            this.label = string.IsNullOrEmpty (label) ? Guid.NewGuid ().ToString () : label;

            var lenWeights = inChannels * kernelSizeX * kernelSizeY * outChannels;

            var vDescWeights = VectorDescriptor (lenWeights);
            weightVectors = new OptimizableVector (device, vDescWeights, 0.0f);

            if (bias) {
                var vDescBiases = VectorDescriptor (outChannels);
                biasVectors = new OptimizableVector (device, vDescBiases, biasInit);
            }
            else {
                biasVectors = null;
            }

            RandomizeWeights ((nuint)DateTime.Now.Ticks, weightsInit);
            //RandomizeGaussianWeights ((nuint)DateTime.Now.Ticks, 0, 0.02f);

            convWtsAndBias = new MPSCnnConvolutionWeightsAndBiasesState (weightVectors.Value.Data, biasVectors?.Value.Data);
            momentumVectors = biasVectors != null ?
                NSArray<MPSVector>.FromNSObjects (weightVectors.Momentum, biasVectors.Momentum) :
                NSArray<MPSVector>.FromNSObjects (weightVectors.Momentum);
            velocityVectors = biasVectors != null ?
                NSArray<MPSVector>.FromNSObjects (weightVectors.Velocity, biasVectors.Velocity) :
                NSArray<MPSVector>.FromNSObjects (weightVectors.Velocity);

            SetOptimizationOptions (true, learningRate: Optimizer.DefaultLearningRate);
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

        public override MPSCnnConvolutionWeightsAndBiasesState Update (IMTLCommandBuffer commandBuffer, MPSCnnConvolutionGradientState gradientState, MPSCnnConvolutionWeightsAndBiasesState sourceState)
        {
            var u = updater;
            if (u != null) {
                updateCount++;

                u.Encode (commandBuffer, gradientState, sourceState, momentumVectors, velocityVectors, convWtsAndBias);

                //Console.WriteLine ($"Update of ConvDataSource {this.Label}");
            }

            return convWtsAndBias;
        }

        //public Dictionary<string, float[]> GetWeights ()
        //{
        //    var r = new Dictionary<string, float[]> {
        //        [label + ".Weights.Value"] = weightVectors.Value.ToArray (),
        //        //[label + ".Weights.Momentum"] = weightVectors.Momentum.ToArray(),
        //        //[label + ".Weights.Velocity"] = weightVectors.Velocity.ToArray(),
        //    };
        //    if (biasVectors != null) {
        //        r[label + ".Biases.Value"] = biasVectors.Value.ToArray ();
        //        //[label + ".Biases.Momentum"] = biasVectors.Momentum.ToArray(),
        //        //[label + ".Biases.Velocity"] = biasVectors.Velocity.ToArray(),
        //    }
        //    return r;
        //}

        void RandomizeWeights (nuint seed, WeightsInit weightsInit)
        {
            var length = weightVectors.Value.Length;
            var a = weightsInit.GetWeights ((int)seed, (int)length);

            weightVectors.Value.SetElements (a);

            weightVectors.Momentum.Zero ();
            weightVectors.Velocity.Zero ();
            biasVectors?.Value.Fill (biasInit);
            biasVectors?.Momentum.Zero ();
            biasVectors?.Velocity.Zero ();

            SetVectorsModified ();
        }

        public bool WeightsAreValid ()
        {
            return weightVectors.IsValid () && (biasVectors != null ? biasVectors.IsValid () : true);
        }

        void SetVectorsModified ()
        {
            weightVectors.Value.Data.DidModify (new NSRange (0, weightVectors.VectorByteSize));
            weightVectors.Momentum.Data.DidModify (new NSRange (0, weightVectors.VectorByteSize));
            weightVectors.Velocity.Data.DidModify (new NSRange (0, weightVectors.VectorByteSize));
            if (biasVectors != null) {
                biasVectors.Value.Data.DidModify (new NSRange (0, biasVectors.VectorByteSize));
                biasVectors.Momentum.Data.DidModify (new NSRange (0, biasVectors.VectorByteSize));
                biasVectors.Velocity.Data.DidModify (new NSRange (0, biasVectors.VectorByteSize));
            }
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

    class OptimizableVector
    {
        public readonly int VectorLength;
        public readonly int VectorByteSize;
        public readonly MPSVectorDescriptor VectorDescriptor;
        public readonly MPSVector Value;
        public readonly MPSVector Momentum;
        public readonly MPSVector Velocity;
        public readonly IntPtr ValuePointer;

        public OptimizableVector (IMTLDevice device, MPSVectorDescriptor descriptor)
        {
            VectorLength = (int)descriptor.Length;
            VectorByteSize = descriptor.GetByteSize ();
            VectorDescriptor = descriptor;
            Value = Vector (device, descriptor, 0.0f);
            Momentum = Vector (device, descriptor, 0.0f);
            Velocity = Vector (device, descriptor, 0.0f);
            ValuePointer = Value.Data.Contents;
        }

        public OptimizableVector (IMTLDevice device, MPSVectorDescriptor descriptor, Tensor initialValue)
            : this (device, descriptor)
        {
            initialValue.Copy (Value.ToSpan (), device);
        }

        public OptimizableVector (IMTLDevice device, MPSVectorDescriptor descriptor, float initialValue)
            : this (device, descriptor)
        {
            Value.Fill (initialValue);
        }

        public void Synchronize (IMTLCommandBuffer commandBuffer)
        {
            Value.Synchronize (commandBuffer);
            Momentum.Synchronize (commandBuffer);
            Velocity.Synchronize (commandBuffer);
        }

        public void DidModify ()
        {
            Value.DidModify ();
            Momentum.DidModify ();
            Velocity.DidModify ();
        }

        public bool IsValid ()
        {
            return Value.IsValid () && Momentum.IsValid () && Velocity.IsValid ();
        }

        public void Zero ()
        {
            Value.Zero ();
            Velocity.Zero ();
            Momentum.Zero ();
        }

#if PB_SERIALIZATION
        public NetworkData.OptimizableVector GetData (bool includeTrainingParameters)
        {
            return new NetworkData.OptimizableVector {
                Value = GetVectorData (Value),
                Momentum = includeTrainingParameters ? GetVectorData (Momentum) : null,
                Velocity = includeTrainingParameters ? GetVectorData (Velocity) : null,
            };
        }

        public void SetData (NetworkData.OptimizableVector data)
        {
            if (data == null)
                return;
            SetVectorData (Value, data.Value);
            SetVectorData (Momentum, data.Momentum);
            SetVectorData (Velocity, data.Velocity);
        }

        NetworkData.Vector GetVectorData (MPSVector vector)
        {
            var v = new NetworkData.Vector ();
            v.Values.AddRange (vector.ToArray ());
            return v;
        }

        bool SetVectorData (MPSVector vector, NetworkData.Vector data)
        {
            if (data == null)
                return false;

            var vs = data.Values;
            var n = (int)vector.Length;
            if (n != vs.Count)
                return false;

            unsafe {
                var p = (float*)vector.Data.Contents;
                for (var i = 0; i < n; i++) {
                    *p++ = vs[i];
                }
            }
            return true;
        }
#endif
    }
}
