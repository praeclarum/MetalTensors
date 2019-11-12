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
    public class Conv2Layer : Layer
    {
        public override int InputCount => 1;

        public int FeatureChannels { get; }
        public int SizeX { get; }
        public int SizeY { get; }
        public int StrideX { get; }
        public int StrideY { get; }
        public ConvPadding Padding { get; }

        public Conv2Layer (int featureChannels, int sizeY, int sizeX, int strideY, int strideX, ConvPadding padding)
        {
            if (featureChannels <= 0)
                throw new ArgumentOutOfRangeException (nameof (featureChannels), "Convolution 2D feature channels must be > 0");

            FeatureChannels = featureChannels;
            SizeX = sizeX;
            SizeY = sizeY;
            StrideX = strideX;
            StrideY = strideY;
            Padding = padding;
        }

        public Conv2Layer (int featureChannels, int size, int stride = 1, ConvPadding padding = ConvPadding.Same)
            : this (featureChannels, size, size, stride, stride, padding)
        {
        }

        public override int[] GetOutputShape (params Tensor[] inputs)
        {
            // https://github.com/keras-team/keras/blob/f06524c44e5f6926968cb2bb3ddd1e523f5474c5/keras/utils/conv_utils.py#L85

            var inputShape = inputs[0].Shape;
            var h = inputShape[0];
            var w = inputShape[1];
            var kh = ConvOutputLength (h, SizeY, StrideY, Padding, 1);
            var kw = ConvOutputLength (w, SizeX, StrideX, Padding, 1);
            //var sh = kh / StrideY;
            //var sw = kw / StrideX;
            return new[] { kh, kw, FeatureChannels };
        }

        static int ConvOutputLength (int inputLength, int size, int stride, ConvPadding padding, int dilation)
        {
            if (inputLength < 0)
                throw new ArgumentOutOfRangeException (nameof (inputLength), "Convolution input dimension must be >= 0");

            var dilatedFilterSize = (size - 1) * dilation + 1;
            var outputLength = padding switch
            {
                ConvPadding.Same => inputLength,
                _ => inputLength - dilatedFilterSize + 1
            };
            var r = (outputLength + stride - 1) / stride;
            return r;
        }

        protected override MPSNNFilterNode CreateFilterNode ((MPSNNImageNode ImageNode, int[] Shape)[] inputs, IMTLDevice device)
        {
            var input = inputs[0];
            int inChannels = input.Shape[2];
            return new MPSCnnConvolutionNode (input.ImageNode, GetWeights (inChannels, device));
        }

        ConvWeights GetWeights (int inChannels, IMTLDevice device)
        {
            var w = new ConvWeights (inChannels, FeatureChannels, SizeY, SizeX, StrideY, StrideX, true, "MEMEMEMEME", device);
            return w;
        }
    }

    public enum ConvPadding
    {
        Same,
        Valid
    }

    class ConvWeights : MPSCnnConvolutionDataSource
    {
        readonly IMTLDevice device;

        readonly string label;
        readonly bool bias;
        readonly MPSCnnConvolutionDescriptor descriptor;

        nuint updateCount;
        MPSNNOptimizerAdam? updater;

        readonly OptimizableVector weightVectors;
        readonly OptimizableVector biasVectors;
        readonly MPSCnnConvolutionWeightsAndBiasesState convWtsAndBias;
        readonly NSArray<MPSVector> momentumVectors;
        readonly NSArray<MPSVector> velocityVectors;

        public override string Label => label;

        public override MPSCnnConvolutionDescriptor Descriptor => descriptor;

        public override MPSDataType DataType => MPSDataType.Float32;

        public override IntPtr Weights => weightVectors.ValuePointer;

        public override IntPtr BiasTerms => biasVectors.ValuePointer;

        public ConvWeights (int inChannels, int outChannels, int kernelSizeY, int kernelSizeX, int strideY, int strideX, bool bias, string label, IMTLDevice device)
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
            this.label = string.IsNullOrEmpty (label) ? Guid.NewGuid ().ToString () : label;

            var lenWeights = inChannels * kernelSizeX * kernelSizeY * outChannels;

            var vDescWeights = VectorDescriptor (lenWeights);
            weightVectors = new OptimizableVector (device, vDescWeights, 0.0f);

            var vDescBiases = VectorDescriptor (outChannels);
            biasVectors = new OptimizableVector (device, vDescBiases, 0.1f);

            convWtsAndBias = new MPSCnnConvolutionWeightsAndBiasesState (weightVectors.Value.Data, biasVectors.Value.Data);
            momentumVectors = NSArray<MPSVector>.FromNSObjects (weightVectors.Momentum, biasVectors.Momentum);
            velocityVectors = NSArray<MPSVector>.FromNSObjects (weightVectors.Velocity, biasVectors.Velocity);

            SetOptimizationOptions (learningRate: 0.0002f);
        }

        void SetOptimizationOptions (float learningRate)
        {
            var odesc = new MPSNNOptimizerDescriptor (learningRate, 1.0f, MPSNNRegularizationType.None, 1.0f);
            updater = new MPSNNOptimizerAdam (
                device,
                beta1: 0.9f, beta2: 0.999f, epsilon: 1e-8f,
                timeStep: 0,
                optimizerDescriptor: odesc);
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

                if (updateCount != u.TimeStep) {
                    throw new Exception ($"Update time step is out of synch");
                }

                //Console.WriteLine ($"UpdateWeights of Conv2dDataSource {this.Label}");
            }

            return convWtsAndBias;
        }

        public Dictionary<string, float[]> GetWeights () => new Dictionary<string, float[]> {
            [label + ".Weights.Value"] = weightVectors.Value.ToArray (),
            //[label + ".Weights.Momentum"] = weightVectors.Momentum.ToArray(),
            //[label + ".Weights.Velocity"] = weightVectors.Velocity.ToArray(),
            [label + ".Biases.Value"] = biasVectors.Value.ToArray (),
            //[label + ".Biases.Momentum"] = biasVectors.Momentum.ToArray(),
            //[label + ".Biases.Velocity"] = biasVectors.Velocity.ToArray(),
        };

        public void RandomizeWeights (nuint seed, IMTLCommandQueue queue)
        {
            var randomDesc = MPSMatrixRandomDistributionDescriptor.CreateUniform (-0.2f, 0.2f);
            var randomKernel = new MPSMatrixRandomMTGP32 (device, MPSDataType.Float32, seed, randomDesc);

            // Run on its own buffer so as not to bother others
            using var commandBuffer = MPSCommandBuffer.Create (queue);
            randomKernel.EncodeToCommandBuffer (commandBuffer, weightVectors.Value);
            commandBuffer.Commit ();
            commandBuffer.WaitUntilCompleted ();

            SetVectorsModified ();
        }

        public bool WeightsAreValid ()
        {
            return weightVectors.WeightsAreValid () && biasVectors.WeightsAreValid ();
        }

        void SetVectorsModified ()
        {
            weightVectors.Value.Data.DidModify (new NSRange (0, weightVectors.VectorByteSize));
            weightVectors.Momentum.Data.DidModify (new NSRange (0, weightVectors.VectorByteSize));
            weightVectors.Velocity.Data.DidModify (new NSRange (0, weightVectors.VectorByteSize));
            biasVectors.Value.Data.DidModify (new NSRange (0, biasVectors.VectorByteSize));
            biasVectors.Momentum.Data.DidModify (new NSRange (0, biasVectors.VectorByteSize));
            biasVectors.Velocity.Data.DidModify (new NSRange (0, biasVectors.VectorByteSize));
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

        public OptimizableVector (IMTLDevice device, MPSVectorDescriptor descriptor, float initialValue)
        {
            VectorLength = (int)descriptor.Length;
            VectorByteSize = descriptor.GetByteSize ();
            VectorDescriptor = descriptor;
            Value = Vector (device, descriptor, initialValue);
            Momentum = Vector (device, descriptor, 0.0f);
            Velocity = Vector (device, descriptor, 0.0f);
            ValuePointer = Value.Data.Contents;
        }

        public void Synchronize (IMTLCommandBuffer commandBuffer)
        {
            Value.Synchronize (commandBuffer);
            Momentum.Synchronize (commandBuffer);
            Velocity.Synchronize (commandBuffer);
        }

        public bool WeightsAreValid ()
        {
            return Value.IsValid () && Momentum.IsValid () && Velocity.IsValid ();
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
