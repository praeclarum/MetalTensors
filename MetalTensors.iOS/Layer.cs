using System;
using System.Linq;
using System.Threading.Tasks;
using Foundation;
using Metal;
using MetalPerformanceShaders;
using MetalTensors.Tensors;

namespace MetalTensors
{
    public abstract class Layer
    {
        public abstract int InputCount { get; }

        public abstract int[] GetOutputShape (params Tensor[] inputs);

        public MPSNNImageNode GetMetalImageNode (Tensor[] inputs, IMTLDevice device)
        {
            var f = GetFilterNode (inputs, device);
            return f.ResultImage;
        }

        public Tensor GetOutput (params Tensor[] inputs)
        {
            return new LayerOutputTensor (this, inputs);
        }

        public Task<Tensor> PredictAsync (Tensor[] inputs, IMTLDevice device)
        {
            if (inputs.Length < InputCount)
                throw new ArgumentException (nameof (inputs));

            var tcs = new TaskCompletionSource<Tensor> ();
            System.Threading.ThreadPool.QueueUserWorkItem (StartGraph);
            return tcs.Task;

            void StartGraph (object s)
            {
                try {
                    var node = GetFilterNode (inputs, device);

                    using var graph = new MPSNNGraph (device, node.ResultImage, true) {
                        Format = MPSImageFeatureChannelFormat.Float32,
                    };
                    //Console.WriteLine (graph.DebugDescription);

                    var sourceHandles = graph.SourceImageHandles;
                    var sources = sourceHandles.Select (x => ((TensorHandle)x).Tensor.GetMetalImage ()).ToArray ();

                    var r = graph.Execute (sources, (image, error) => {
                        if (error != null) {
                            tcs.SetException (new Exception (error.Description));
                        }
                        else {
                            var t = new MPSImageTensor (image);
                            tcs.SetResult (t);
                        }
                    });
                }
                catch (Exception ex) {
                    tcs.TrySetException (ex);
                }
            }
        }

        static IMTLDevice? FindDevice (Tensor[] tensors)
        {
            // TODO: Scan inputs for the correct device to use
            return null;
        }

        protected MPSNNFilterNode GetFilterNode (Tensor[] inputs, IMTLDevice device)
        {
            var inputImageNodes = inputs.Select (x => (x.GetMetalImageNode (device), x.Shape)).ToArray ();
            var node = CreateFilterNode (inputImageNodes, device);
            return node;
        }

        protected abstract MPSNNFilterNode CreateFilterNode ((MPSNNImageNode ImageNode, int[] Shape)[] inputs, IMTLDevice device);
    }
}
