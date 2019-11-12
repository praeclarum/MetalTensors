using System;
using System.Linq;
using MetalPerformanceShaders;

namespace MetalTensors.Layers
{
    public class AddLayer : Layer
    {
        public override int[] GetOutputShape (params Tensor[] inputs)
        {
            return inputs[0].Shape;
        }

        public override Tensor Compute (Tensor[] inputs)
        {
            if (inputs.Length != 2)
                throw new ArgumentException (nameof (inputs));

            var inputImageNodes = inputs.Select (x => x.ToImageNode ()).ToArray ();

            var node = new MPSNNAdditionNode (inputImageNodes);
            Console.WriteLine (node.DebugDescription);

            var device = MetalExtensions.Current (null);
            var graph = new MPSNNGraph (device, node.ResultImage, true);
            Console.WriteLine (graph.DebugDescription);

            var sourceHandles = graph.SourceImageHandles;
            Console.WriteLine (sourceHandles);
            var sourceImages = 0;

            //var result = graph.Execute (sourceImages, null);
            //Console.WriteLine (result.DebugDescription);

            throw new NotImplementedException ();
        }
    }
}
