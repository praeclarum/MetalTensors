using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Foundation;
using Metal;
using MetalPerformanceShaders;

using static MetalTensors.MetalHelpers;

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

        int updateCount;
        MPSNNOptimizerAdam? optimizer;
        bool trainable;

        Lazy<ConvWeightValues> convWeights;
        readonly object convWeightsMutex = new object ();
        readonly Func<ConvWeightValues> createWeightValues;
        readonly MPSCnnConvolutionDescriptor descriptor;

        public override string Label => label;

        /// <summary>
        /// For convolution, MPSDataType.UInt8, MPSDataType.Float16, and MPSDataType.Float32 are supported.
        /// </summary>
        public override MPSDataType DataType => MPSDataType.Float32;

        /// <summary>
        /// The type of each entry in array is given by -dataType. The number of entries is equal to:
        ///     inputFeatureChannels * outputFeatureChannels* kernelHeight * kernelWidth
        /// The layout of filter weight is as a 4D tensor (array):
        ///     weight[outputChannels][kernelHeight][kernelWidth][inputChannels / groups]
        /// </summary>
        public override IntPtr Weights => convWeights.Value.WeightVectors.ValuePointer;

        /// <summary>
        /// Each entry in the array is a single precision IEEE-754 float and represents one bias.
        /// The number of entries is equal to outputFeatureChannels.
        /// Note: bias terms are always float, even when the weights are not.
        /// </summary>
        public override IntPtr BiasTerms => convWeights.Value.BiasVectors is OptimizableVector v ? v.ValuePointer : IntPtr.Zero;

        public override MPSCnnConvolutionDescriptor Descriptor => descriptor;

        //readonly IMTLCommandQueue weightsQueue;


        public ConvDataSource (ConvWeights convWeights, IMTLDevice device)
            : this (convWeights.InChannels, convWeights.OutChannels,
                    convWeights.SizeX, convWeights.SizeY, convWeights.StrideX, convWeights.StrideY,
                    convWeights.Bias, convWeights.WeightsInit, convWeights.BiasInit,
                    convWeights.Label, device)
        {
        }

        public ConvDataSource (int inChannels, int outChannels, int kernelSizeX, int kernelSizeY, int strideX, int strideY, bool bias, WeightsInit weightsInit, float biasInit, string label, IMTLDevice device)
        {
            if (inChannels <= 0)
                throw new ArgumentOutOfRangeException (nameof (inChannels), "Number of convolution input channels must be > 0");
            if (outChannels <= 0)
                throw new ArgumentOutOfRangeException (nameof (inChannels), "Number of convolution output channels must be > 0");

            this.device = device;

            //var queue = device.CreateCommandQueue ();
            //if (queue == null)
            //    throw new Exception ($"Failed to create queue to load values");
            //weightsQueue = queue;

            descriptor = MPSCnnConvolutionDescriptor.CreateCnnConvolutionDescriptor (
                (System.nuint)kernelSizeX, (System.nuint)kernelSizeY,
                (System.nuint)inChannels,
                (System.nuint)outChannels);
            descriptor.StrideInPixelsX = (nuint)strideX;
            descriptor.StrideInPixelsY = (nuint)strideY;

            createWeightValues = () => {
                //Console.WriteLine ($"CREATING WEIGHT VALUES {label}");
                return new ConvWeightValues (inChannels, outChannels, kernelSizeX, kernelSizeY, bias, weightsInit, biasInit, device);
            };
            convWeights = new Lazy<ConvWeightValues> (createWeightValues);

            this.bias = bias;
            this.biasInit = biasInit;
            this.label = string.IsNullOrEmpty (label) ? Guid.NewGuid ().ToString () : label;
            this.trainable = false; // SetOptimizationOptions needs to be called in order to Update
        }

        public void SetOptimizationOptions (bool trainable, float learningRate)
        {
            this.trainable = trainable;
            if (trainable) {
                if (optimizer is MPSNNOptimizerAdam adam) {
                    adam.SetLearningRate (learningRate);
                }
                else {
                    var odesc = new MPSNNOptimizerDescriptor (learningRate, 1.0f, MPSNNRegularizationType.None, 1.0f);
                    optimizer = new MPSNNOptimizerAdam (
                        device,
                        beta1: 0.9f, beta2: 0.999f, epsilon: 1e-7f,
                        timeStep: 0,
                        optimizerDescriptor: odesc);
                }
            }
        }

        /// <summary>
        /// Alerts the data source that the data will be needed soon.
        /// 
        /// Each load alert will be balanced by a purge later, when MPS
        /// no longer needs the data from this object.
        /// Load will always be called atleast once after initial construction
        /// or each purge of the object before anything else is called.
        /// Note: load may be called to merely inspect the descriptor.
        /// 
        /// In some circumstances, it may be worthwhile to postpone
        /// weight and bias construction until they are actually needed
        /// to save touching memory and keep the working set small.
        /// The load function is intended to be an opportunity to open
        /// files or mark memory no longer purgeable.
        /// </summary>
        [DebuggerHidden]
        public override bool Load {
            get {
                // convWeights is always ready to load data. Even after a Purge().
                // Don't force its value here because sometimes Load is called
                // just to get the descriptor :-(
                var cw = convWeights;                
                if (cw.IsValueCreated) {
                    var wv = cw.Value;
                    var wtsB = wv.ConvWtsAndBias;
                    using var pool = new NSAutoreleasePool ();
                    using var queue = device.CreateCommandQueue ();
                    if (queue == null)
                        throw new Exception ($"Failed to create queue to load values");
                    var commands = MPSCommandBuffer.Create (queue);
                    wtsB.Synchronize (commands);
                    commands.Commit ();
                    commands.WaitUntilCompleted ();
                }
                return true;
            }
        }

        /// <summary>
        /// Alerts the data source that the data is no longer needed.
        /// Each load alert will be balanced by a purge later, when MPS
        /// no longer needs the data from this object.
        /// </summary>
        public override void Purge ()
        {
            try {
                // DONT PURGE UNTIL WE HAVE A WAY TO STORE WEIGHTS
                //Console.WriteLine ($"PURGING WEIGHT VALUES {label}");
                //ConvWeightValues? oldWeights = null;
                //lock (convWeightsMutex) {
                //    //Console.WriteLine ($"Purge Conv2dDataSource {this.Label}");
                //    if (convWeights.IsValueCreated) {
                //        oldWeights = convWeights.Value;
                //        convWeights = new Lazy<ConvWeightValues> (createWeightValues);
                //    }
                //}
                //oldWeights?.Dispose ();
            }
            catch (Exception ex) {
                Console.WriteLine ($"Failed to purge weights: {ex}");
            }
        }

        /// <summary>
        /// Callback for the MPSNNGraph to update the convolution weights on GPU.
        ///
        /// It is the resposibility of this method to decrement the read count of both the gradientState and the sourceState before returning.
        /// </summary>
        /// <param name="commandBuffer">The command buffer on which to do the update.
        /// MPSCNNConvolutionGradientNode.MPSNNTrainingStyle controls where you want your update
        /// to happen. Provide implementation of this function for GPU side update.</param>
        /// <param name="gradientState">A state object produced by the MPSCNNConvolution and updated by MPSCNNConvolutionGradient containing weight gradients.</param>
        /// <param name="sourceState">A state object containing the convolution weights.</param>
        /// <returns>If NULL, no update occurs. If nonnull, the result will be used to update the weights in the MPSNNGraph</returns>
        public override MPSCnnConvolutionWeightsAndBiasesState? Update (IMTLCommandBuffer commandBuffer, MPSCnnConvolutionGradientState gradientState, MPSCnnConvolutionWeightsAndBiasesState sourceState)
        {
            if (trainable) {
                var v = convWeights.Value;
                var opt = optimizer;
                if (opt != null) {
                    Interlocked.Increment (ref updateCount);

                    opt.Encode (commandBuffer, gradientState, sourceState, v.momentumVectors, v.velocityVectors, v.ConvWtsAndBias);

                    //Console.WriteLine ($"Update of ConvDataSource {this.Label}");
                }
                else {
                    throw new Exception ($"Attempted to Update without an Optimizer");
                }
                return v.ConvWtsAndBias;
            }
            else {
                return null;
            }
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

    /// <summary>
    /// This is a separate class from the DataSource so we can delay-load the values.
    /// </summary>
    sealed class ConvWeightValues : IDisposable
    {
        readonly OptimizableVector weightVectors;
        public OptimizableVector WeightVectors {
            get {
                weightInitTask.Wait ();
                return weightVectors;
            }
        }
        public readonly OptimizableVector? BiasVectors;
        MPSCnnConvolutionWeightsAndBiasesState? convWtsAndBias;
        public MPSCnnConvolutionWeightsAndBiasesState ConvWtsAndBias {
            get {
                weightInitTask.Wait ();
                return convWtsAndBias!;
            }
        }
        public readonly NSArray<MPSVector> momentumVectors;
        public readonly NSArray<MPSVector> velocityVectors;
        private bool disposed;
        private Task weightInitTask;

        public ConvWeightValues (int inChannels, int outChannels, int kernelSizeX, int kernelSizeY, bool bias, WeightsInit weightsInit, float biasInit, IMTLDevice device)
        {
            var lenWeights = inChannels * kernelSizeX * kernelSizeY * outChannels;

            var vDescWeights = VectorDescriptor (lenWeights);
            weightVectors = new OptimizableVector (device, vDescWeights, 0.0f);

            if (bias) {
                var vDescBiases = VectorDescriptor (outChannels);
                BiasVectors = new OptimizableVector (device, vDescBiases, biasInit);
            }
            else {
                BiasVectors = null;
            }

            momentumVectors = BiasVectors != null ?
                NSArray<MPSVector>.FromNSObjects (weightVectors.Momentum, BiasVectors.Momentum) :
                NSArray<MPSVector>.FromNSObjects (weightVectors.Momentum);
            velocityVectors = BiasVectors != null ?
                NSArray<MPSVector>.FromNSObjects (weightVectors.Velocity, BiasVectors.Velocity) :
                NSArray<MPSVector>.FromNSObjects (weightVectors.Velocity);

            weightInitTask = Task.Run (async () => {
                var seed = (int)DateTime.Now.Ticks;
                var fanIn = inChannels * kernelSizeX * kernelSizeY;
                var fanOut = outChannels * kernelSizeX * kernelSizeY;
                await weightsInit.InitWeightsAsync (weightVectors.Value, seed, fanIn, fanOut).ConfigureAwait (false);
                convWtsAndBias = new MPSCnnConvolutionWeightsAndBiasesState (weightVectors.Value.Data, BiasVectors?.Value.Data);
            });
        }

        public void Dispose ()
        {
            if (!disposed) {
                disposed = true;
                velocityVectors.Dispose ();
                momentumVectors.Dispose ();
                ConvWtsAndBias.Dispose ();
                BiasVectors?.Dispose ();
                WeightVectors?.Dispose ();
            }
        }
    }

    sealed class OptimizableVector : IDisposable
    {
        public readonly int VectorLength;
        public readonly int VectorByteSize;
        public readonly MPSVectorDescriptor VectorDescriptor;
        public readonly MPSVector Value;
        public readonly MPSVector Momentum;
        public readonly MPSVector Velocity;
        public readonly IntPtr ValuePointer;
        private bool disposed;

        /// <summary>
        /// Momentum and Velocity are initialized to 0. Value is left uninitialized.
        /// </summary>
        public OptimizableVector (IMTLDevice device, MPSVectorDescriptor descriptor)
        {
            VectorLength = (int)descriptor.Length;
            VectorByteSize = descriptor.GetByteSize ();
            VectorDescriptor = descriptor;
            Value = Vector (descriptor, device);
            Momentum = Vector (0.0f, descriptor, device);
            Velocity = Vector (0.0f, descriptor, device);
            ValuePointer = Value.Data.Contents;
        }

        /// <summary>
        /// Momentum and Velocity are initialized to 0. Value is copied from the tensor.
        /// </summary>
        public OptimizableVector (IMTLDevice device, MPSVectorDescriptor descriptor, Tensor initialValue)
            : this (device, descriptor)
        {
            initialValue.Copy (Value.ToSpan (), device);
        }

        /// <summary>
        /// Momentum and Velocity are initialized to 0. Value is filled with a constant.
        /// </summary>
        public OptimizableVector (IMTLDevice device, MPSVectorDescriptor descriptor, float initialValue)
            : this (device, descriptor)
        {
            Value.Fill (initialValue);
        }

        public void Dispose ()
        {
            if (!disposed) {
                disposed = true;
                Velocity.Dispose ();
                Momentum.Dispose ();
                Value.Dispose ();
            }
        }

        /// <summary>
        /// Flush the underlying MTLBuffer from the device's caches, and invalidate any CPU caches if needed.
        /// This will call[id < MTLBlitEncoder > synchronizeResource: ] on the vector's MTLBuffer, if any.
        /// This is necessary for all MTLStorageModeManaged resources.For other resources, including temporary
        /// resources (these are all MTLStorageModePrivate), and buffers that have not yet been allocated, nothing is done.
        /// It is more efficient to use this method than to attempt to do this yourself with the data property.
        /// </summary>
        /// <param name="commandBuffer"></param>
        public void DownloadFromGpu (IMTLCommandBuffer commandBuffer)
        {
            Value.Synchronize (commandBuffer);
            Momentum.Synchronize (commandBuffer);
            Velocity.Synchronize (commandBuffer);
        }


        /// <summary>
        /// Informs the GPU that the CPU has modified the vectors.
        /// </summary>
        public void MarkAsModifiedByCpu ()
        {
            Value.MarkAsModifiedByCpu ();
            Momentum.MarkAsModifiedByCpu ();
            Velocity.MarkAsModifiedByCpu ();
        }

        public bool IsFinite ()
        {
            return Value.IsFinite () && Momentum.IsFinite () && Velocity.IsFinite ();
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
