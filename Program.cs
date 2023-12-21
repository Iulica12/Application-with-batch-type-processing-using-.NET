using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.Spark.Sql;
using Microsoft.Spark.Sql.Streaming;
class program
{
    static void Main(string[] args)
    {
        SparkSession spark = SparkSession.Builder().AppName("Streaming Review Analysis").GetOrCreate();

        string hostname = "localhost";
        int port = 9999;

        DataFrame words = spark.ReadStream().Format("socket").Option("host", hostname).Option("port", port).Load();

        spark.Udf().Register<string, bool>("MLudf", input => Sentiment(input, args[0]));

        words.CreateOrReplaceTempView("WordsSentiment"); DataFrame sqlDf = spark.Sql("SELECT WordsSentiment.value, MLudf(WordsSentiment.value) FROM WordsSentiment");
        // Gestionam datele de intrare in mod contiunuu pe masura ce acestea sunt transmise
        StreamingQuery query = sqlDf.WriteStream().Format("console").Start();
        query.AwaitTermination();
    }

    static bool Sentiment(string text, string modelPath)
    {
        var mlContext = new MLContext();
        ITransformer mlModel = mlContext.Model.Load(modelPath, out var modelInputSchema);
        PredictionEngine<Review, ReviewPrediction> predEngine = mlContext.Model.CreatePredictionEngine<Review, ReviewPrediction>(mlModel);
        ReviewPrediction result = predEngine.Predict(new Review { ReviewText = text });

        return result.Prediction;
    }
}
public class Review { [LoadColumn(0)] public string ReviewText; }
public class ReviewPrediction : Review
{
    [ColumnName("PredictedLabel")]
    public bool Prediction { get; set; }
    public float Probability { get; set; }
    public float Score { get; set; }
}