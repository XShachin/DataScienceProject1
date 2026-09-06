import scalation.mathstat._
import scalation.modeling.SimpleRegression

@main def AutoMPG(): Unit = 
    // Load cleaned AutoMPG data
    val data = MatrixD.load("../data/auto_mpg_clean.csv", skip = 1)

    // Select MPG as response
    val y = data.col(0)

    // Predictors: displacement, weight
    val displacement = data.col(2)
    val weight = data.col(4)

    // SimpleRegression for MPG vs Displacement
    println("\n Simple Regression: MPG vs Displacement")

    val displacementModel =
        SimpleRegression(displacement, y, Array("displacement", "mpg"))

    displacementModel.train()
    val (displacementPredictions, displacementQof) = displacementModel.test()
    println(displacementModel.report(displacementQof))
    println(displacementModel.summary())

    println("Interpretation: MPG decreases as displacement increases, showing a negative correlation. " +
        "The model explains about 64.8% of the variation in MPG.")

    new Plot(
        displacement,
        y,
        displacementPredictions,
        "Observed vs Fitted MPG by Displacement",
        lines = true
    )

    // SimpleRegression for MPG vs Weight
    println("\n Simple Regression: MPG vs Weight")

    val weightModel =
        SimpleRegression(weight, y, Array("weight", "mpg"))

    weightModel.train()
    val (weightPredictions, weightQof) = weightModel.test()
    println(weightModel.report(weightQof))
    println(weightModel.summary())

    println("Interpretation: MPG decreases as vehicle weight increases, showing a negative correlation. " +
        "The model explains about 69.3% of the variation in MPG.")

    new Plot(
        weight,
        y,
        weightPredictions,
        "Observed vs Fitted MPG by Weight",
        lines = true
    )

