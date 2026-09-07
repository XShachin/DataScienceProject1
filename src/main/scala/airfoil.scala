import scalation.mathstat.MatrixD
import scalation.modeling.SimpleRegression

@main def Airfoil(): Unit = 
    // Load cleaned AutoMPG data
    val data = MatrixD.load("../data/airfoil_self_noise.csv", skip = 1)

    // Select MPG as response
    val y = data.col(5)

    // Predictors: displacement, weight
    val frequency = data.col(0)
    val displacement = data.col(4)

    // SimpleRegression for MPG vs Displacement
    println("\n Simple Regression: Sound Pressure vs Frequency")

    val frequencyModel =
        SimpleRegression(frequency, y, Array("frequency", "Sound_Pressure_Level"))

    frequencyModel.train()
    val (_, frequencyQof) = frequencyModel.test()
    println(frequencyModel.report(frequencyQof))
    println(frequencyModel.summary())

    println("Interpretation: Sound Pressure and Frequency have an inverse relationship, " +
      "where if one increases the other decreases. With around .0009 dB decrease in the Sound pressure" +
      " for every Hz increase in frequency")

    // SimpleRegression for MPG vs Weight
    println("\n Simple Regression: Sound Pressure vs Displacement Thickness")

    val displacementModel =
        SimpleRegression(displacement, y, Array("Displacement_Thickness", "Sound_Pressure_Level"))

    displacementModel.train()
    val (_, displacementQof) = displacementModel.test()
    println(displacementModel.report(displacementQof))
    println(displacementModel.summary())

    println("Interpretation: Sound Pressure and Displacement Thickness have an inverse relationship, " +
      "where if one increases the other decreses. With around 164.027 dB decrease in Sound pressure" +
      " for every mm increase in displacement thickness")