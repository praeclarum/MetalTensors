namespace MetalTensors
{
    public class TrainingHistory
    {
        public Tensor[][] Losses { get; }

        public TrainingHistory (Tensor[][] losses)
        {
            Losses = losses;
        }
    }
}
