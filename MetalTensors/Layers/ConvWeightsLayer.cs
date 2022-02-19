using System;
using System.Collections.Concurrent;
using Metal;
using MetalPerformanceShaders;

namespace MetalTensors.Layers
{
    public abstract class ConvWeightsLayer : Layer
    {
        public override int MinInputCount => 1;

        public int InFeatureChannels => Weights.InChannels;
        public int OutFeatureChannels => Weights.OutChannels;
        public int SizeX => Weights.SizeX;
        public int SizeY => Weights.SizeY;
        public int StrideX => Weights.StrideX;
        public int StrideY => Weights.StrideY;
        public bool Bias => Weights.Bias;
        public WeightsInit WeightsInit => Weights.WeightsInit;
        public float BiasInit => Weights.BiasInit;
        public ConvPadding Padding { get; }

        public ConvWeights Weights { get; }

        protected ConvWeightsLayer (int inFeatureChannels, int outFeatureChannels, int sizeX, int sizeY, int strideX, int strideY, ConvPadding padding, bool bias, WeightsInit weightsInit, float biasInit, string? name = null, bool isTrainable = true)
            : base (name, isTrainable: isTrainable)
        {
            Weights = new ConvWeights (Name, inFeatureChannels, outFeatureChannels, sizeX, sizeY, strideX, strideY, bias, weightsInit, biasInit);
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
            return CreateConvWeightsNode (inputs[0].ImageNode, GetMetalConvDataSource (device));
        }

        public override MPSCnnConvolutionDataSource GetMetalConvDataSource (IMTLDevice device)
        {
            return Weights.GetDataSource (device);
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
}
