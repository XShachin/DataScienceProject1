import scalation.modeling.SimpleRegression
import scalation.mathstat._

@main def Concrete(): Unit = 
    // Load cleaned Concrete data
    val data = MatrixD.load("../data/Concrete_Data.csv", skip = 1)

    // Select Strength as response
    val y = data.col(8)

    // Predictors: Cement, Superplasticizer
    val cement = data.col(0)
    val superplasticizer = data.col(4)

    // SimpleRegression for Strength vs Cement
    println("\n Simple Regression: Strength vs Cement")

    val cementModel =
        SimpleRegression(cement, y, Array("cement", "strength"))

    cementModel.train()
    val (cementPredictions, cementQof) = cementModel.test()
    println(cementModel.report(cementQof))
    println(cementModel.summary())

    println("Interpretation: Strength has a positive, linear correlation with" +
        " cement content. The model explains about 24.8% of the variation in Strength.")

    new Plot(
        cement,
        y,
        cementPredictions,
        "Observed vs Fitted Strength by Cement",
        lines = true
    )

    // SimpleRegression for Strength vs Superplasticizer
    println("\n Simple Regression: Strength vs Superplasticizer")
    val superplasticizerModel = 
        SimpleRegression(superplasticizer, y, Array("superplasticizer", "strength"))

    superplasticizerModel.train()
    val (superplasticizerPredictions, superplasticizerQof) = superplasticizerModel.test()
    println(superplasticizerModel.report(superplasticizerQof))
    println(superplasticizerModel.summary())

    println("Interpretation: Strength has a positive, linear correlation with" +
        " superplasticizer content. The model explains about 13.4% of the variation in Strength.")
    new Plot(
        superplasticizer,
        y,
        superplasticizerPredictions,
        "Observed vs Fitted Strength by Superplasticizer",
        lines = true
    )