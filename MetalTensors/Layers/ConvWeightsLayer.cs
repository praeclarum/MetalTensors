using System;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using Foundation;
using Metal;
using MetalPerformanceShaders;

using static MetalTensors.MetalHelpers;

namespace MetalTensors.Layers
{
    public abstract class ConvWeightsLayer : WeightsLayer
    {
        public override int MinInputCount => 1;

        public int InFeatureChannels { get; }
        public int OutFeatureChannels { get; }
        public int SizeX { get; }
        public int SizeY { get; }
        public int StrideX { get; }
        public int StrideY { get; }
        public ConvPadding Padding { get; }
        public bool Bias { get; }
        public WeightsInit WeightsInit { get; }
        public float BiasInit { get; }

        public override int ParameterCount => InFeatureChannels * OutFeatureChannels * SizeX * SizeY + (Bias ? OutFeatureChannels : 0);

        protected ConvWeightsLayer (
            int inFeatureChannels,
            int outFeatureChannels,
            int sizeX,
            int sizeY,
            int strideX,
            int strideY,
            ConvPadding padding,
            bool bias,
            WeightsInit weightsInit,
            float biasInit,
            string? name,
            bool isTrainable)
            : base (name, isTrainable: isTrainable, bias ? new[] { "weights", "bias" } : new[] { "weights" })
        {
            if (inFeatureChannels <= 0)
                throw new ArgumentOutOfRangeException (nameof (inFeatureChannels), "Number of convolution input channels must be > 0");
            if (outFeatureChannels <= 0)
                throw new ArgumentOutOfRangeException (nameof (outFeatureChannels), "Number of convolution output channels must be > 0");

            InFeatureChannels = inFeatureChannels;
            OutFeatureChannels = outFeatureChannels;
            SizeX = sizeX;
            SizeY = sizeY;
            StrideX = strideX;
            StrideY = strideY;
            Bias = bias;
            WeightsInit = weightsInit;
            BiasInit = biasInit;
            Padding = padding;
        }

        public override Config Config => base.Config.Update (new Config {
            { "inFeatureChannels", InFeatureChannels },
            { "outFeatureChannels", OutFeatureChannels },
            { "sizeX", SizeX },
            { "sizeY", SizeY },
            { "strideX", StrideX },
            { "strideY", StrideY },
            { "padding", Padding },
            { "bias", Bias },
            { "biasInit", BiasInit },
            { "weightsInit", WeightsInit },
        });

        protected override MPSNNFilterNode CreateFilterNode ((MPSNNImageNode ImageNode, int[] Shape)[] inputs, IMTLDevice device)
        {
            return CreateConvWeightsNode (inputs[0].ImageNode, GetDataSource<ConvDataSource> (device));
        }

        protected override IWeightsDataSource CreateDataSource (IMTLCommandQueue queue)
        {
            return new ConvDataSource (this, queue);
        }

        protected abstract MPSNNFilterNode CreateConvWeightsNode (MPSNNImageNode imageNode, MPSCnnConvolutionDataSource convDataSource);

        public static int ConvOutputLength (int inputLength, int size, int stride, ConvPadding padding, int dilation)
        {
            // https://github.com/keras-team/keras/blob/afff7b4326f380a54c73400d1e2ae03890162bdf/keras/utils/conv_utils.py#L85

            if (inputLength < 0)
                throw new ArgumentOutOfRangeException (nameof (inputLength), "Conv input dimension must be >= 0");

            var dilatedFilterSize = (size - 1) * dilation + 1;
            var outputLength = padding switch
            {
                ConvPadding.Same => inputLength,
                _ => inputLength - dilatedFilterSize + 1
            };
            var r = (outputLength + stride - 1) / stride;
            return r;
        }

        //static int DestSizeReverse (int sourceSize, int stride, int filterWindowSize, Style style)
        //{
        //    // style = {-1,0,1} for valid-only, same, full
        //    return (sourceSize - 1) * stride + 1 + style * (filterWindowSize - 1);
        //}
        public static int ConvTransposeOutputLength (int inputLength, int size, int stride, ConvPadding padding, int dilation, int? outputPadding)
        {
            // https://github.com/keras-team/keras/blob/b75b2f7dcf5d3c83e33b8b2bc86f1d2543263a59/keras/utils/conv_utils.py#L138

            if (inputLength < 0)
                throw new ArgumentOutOfRangeException (nameof (inputLength), "Conv transpose input dimension must be >= 0");

            var kernel_size = (size - 1) * dilation + 1;

            var dim_size = inputLength;

            if (outputPadding == null) {
                switch (padding) {
                    case ConvPadding.Same:
                        dim_size = dim_size * stride;
                        break;
                    default:
                    case ConvPadding.Valid:
                        dim_size = dim_size * stride + Math.Max (kernel_size - stride, 0);
                        break;
                }
            }
            else {
                int pad;
                switch (padding) {
                    case ConvPadding.Same:
                        pad = kernel_size / 2;
                        break;
                    default:
                    case ConvPadding.Valid:
                        pad = 0;
                        break;
                }
                dim_size = (dim_size - 1) * stride + kernel_size - 2 * pad + outputPadding.Value;
            }

            return dim_size;
        }
    }

    class ConvDataSource : MPSCnnConvolutionDataSource, IWeightsDataSource
    {
        readonly ConvWeightsLayer layer;
        readonly IMTLCommandQueue weightsQueue;
        readonly IMTLDevice device;

        int updateCount;
        int loadedUpdateCount;
        MPSNNOptimizerAdam? optimizer;
        bool trainable; // SetOptimizationOptions needs to be called in order to Update

        Lazy<ConvWeightValues> convWeights;
        readonly Func<ConvWeightValues> createWeightValues;
        readonly MPSCnnConvolutionDescriptor descriptor;

        public override string Label => layer.Name;

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

        public ConvDataSource (ConvWeightsLayer layer, IMTLCommandQueue weightsQueue)
        {
            this.layer = layer;
            this.weightsQueue = weightsQueue;
            device = weightsQueue.Device;

            descriptor = MPSCnnConvolutionDescriptor.CreateCnnConvolutionDescriptor (
                (System.nuint)layer.SizeX, (System.nuint)layer.SizeY,
                (System.nuint)layer.InFeatureChannels,
                (System.nuint)layer.OutFeatureChannels);
            descriptor.StrideInPixelsX = (nuint)layer.StrideX;
            descriptor.StrideInPixelsY = (nuint)layer.StrideY;

            createWeightValues = () => {
                //Console.WriteLine ($"CREATING WEIGHT VALUES {label}");
                return new ConvWeightValues (layer, weightsQueue);
            };
            convWeights = new Lazy<ConvWeightValues> (createWeightValues);
        }

        public void SetOptimizationOptions (bool trainable, Optimizer newOptimizer)
        {
            this.trainable = trainable;
            if (trainable) {
                if (newOptimizer is AdamOptimizer newAdam) {
                    if (optimizer is MPSNNOptimizerAdam adam) {
                        adam.SetLearningRate (newOptimizer.LearningRate);
                    }
                    else {
                        var odesc = new MPSNNOptimizerDescriptor (newOptimizer.LearningRate, 1.0f, MPSNNRegularizationType.None, 1.0f);
                        optimizer = new MPSNNOptimizerAdam (
                            device,
                            beta1: newAdam.Beta1, beta2: newAdam.Beta2, epsilon: newAdam.Epsilon,
                            timeStep: 0,
                            optimizerDescriptor: odesc);
                    }
                }
                else {
                    throw new NotSupportedException (newOptimizer.GetType().Name + " is not supported");
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
                    if (updateCount != loadedUpdateCount) {
                        loadedUpdateCount = updateCount;
                        using var pool = new NSAutoreleasePool ();
                        var wv = cw.Value;
                        var wtsB = wv.ConvWtsAndBias;
                        var commands = MPSCommandBuffer.Create (weightsQueue);
                        wtsB.Synchronize (commands);
                        commands.Commit ();
                        commands.WaitUntilCompleted ();
                    }
                }
                return true;
            }
        }

        public bool DownloadWeightsFromGpu ()
        {
            return Load;
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

                    //Console.WriteLine ($"Update of ConvDataSource {this.Label}");
                    opt.Encode (commandBuffer, gradientState, sourceState, v.momentumVectors, v.velocityVectors, v.ConvWtsAndBias);

                    Interlocked.Increment (ref updateCount);
                }
                else {
                    throw new InvalidOperationException ($"Attempted to Update ConvWeightsLayer without an Optimizer");
                }
                return v.ConvWtsAndBias;
            }
            else {
                gradientState.ReadCount--;
                sourceState.ReadCount--;
                return null;
            }
        }
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

        public ConvWeightValues (ConvWeightsLayer layer, IMTLCommandQueue queue)
        {
            var inChannels = layer.InFeatureChannels;
            var outChannels = layer.OutFeatureChannels;
            var kernelSizeX = layer.SizeX;
            var kernelSizeY = layer.SizeY;
            var bias = layer.Bias;
            var biasInit = layer.BiasInit;
            var device = queue.Device;

            var lenWeights = inChannels * kernelSizeX * kernelSizeY * outChannels;

            var vDescWeights = VectorDescriptor (lenWeights);
            weightVectors = new OptimizableVector (device, vDescWeights);

            if (bias) {
                var vDescBiases = VectorDescriptor (outChannels);
                BiasVectors = new OptimizableVector (device, vDescBiases);
                layer.AddParameter ("bias", BiasVectors, layer.BiasInit);
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
                var fanIn = inChannels * kernelSizeX * kernelSizeY;
                var fanOut = outChannels * kernelSizeX * kernelSizeY;
                await layer.AddParameterAsync ("weights", weightVectors, layer.WeightsInit, fanIn, fanOut, queue).ConfigureAwait (false);
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
}
