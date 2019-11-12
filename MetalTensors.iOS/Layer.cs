using System;
using System.Linq;
using System.Threading.Tasks;
using Foundation;
using MetalPerformanceShaders;
using MetalTensors.Tensors;

namespace MetalTensors
{
    public abstract class Layer
    {
        public abstract int InputCount { get; }

        public abstract int[] GetOutputShape (params Tensor[] inputs);

        public Tensor GetOutput (params Tensor[] inputs)
        {
            return new LayerOutputTensor (this, inputs);
        }

        public Task<Tensor> PredictAsync (Tensor[] inputs)
        {
            if (inputs.Length != InputCount)
                throw new ArgumentException (nameof (inputs));

            var inputImageNodes = inputs.Select (x => x.ImageNode).ToArray ();
            var node = CreateFilterNode (inputImageNodes);

            var device = MetalExtensions.Current (null);
            using var graph = new MPSNNGraph (device, node.ResultImage, true) {
                Format = MPSImageFeatureChannelFormat.Float32,
            };
            //Console.WriteLine (graph.DebugDescription);

            var sourceHandles = graph.SourceImageHandles;
            var sources = sourceHandles.Select (x => ((TensorHandle)x).Tensor.GetImage ()).ToArray ();

            var tcs = new TaskCompletionSource<Tensor> ();
            graph.Execute (sources, (image, error) => {
                if (error != null) {
                    tcs.SetException (new Exception (error.Description));
                }
                else {
                    var t = new MPSImageTensor (image);
                    tcs.SetResult (t);
                }
            });

            return tcs.Task;
        }

        protected abstract MPSNNFilterNode CreateFilterNode (MPSNNImageNode[] inputImageNodes);
    }
}
