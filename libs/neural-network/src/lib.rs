pub struct Network {
    layers: Vec<Layer>,
}

struct Layer {
    neurons: Vec<Neuron>,
}

struct Neuron {
    bias: f32,
    weights: Vec<f32>,
}

impl Network {
    pub fn propagate(&self, mut inputs: Vec<f32>) -> Vec<f32> {
        //send the input to the next layer, where it processes it, after looping through all layers, return input
        for layer in &self.layers {
            inputs = layer.propagate(inputs);
        }

        inputs
    }
}

impl Layer {
    fn propagate(&self, mut inputs: Vec<f32>) -> Vec<f32> {
        //loops over each neuron in the layer, gives it the input, and expects a f32 from it
        let mut outputs = Vec::with_capacity(self.neurons.len());

        for neuron in &self.neurons {
            let output = neuron.propagate(&inputs);
            outputs.push(output);
        }
        outputs
    }
}

impl Neuron {
    fn propagate(&self, inputs: &[f32]) -> f32 {
        //loops over all the input values, multiplies it by the given weight, and then adds bias
        let output = inputs
            .iter()
            .zip(&self.weights)
            .map(|(input, weight)| input * weight)
            .sum::<f32>();

        (self.bias + output).max(0.0)

    }
}

//neuron loops over last layers output, multiplies it by the weights, then adds the bias
//returns it and adds it to a vector

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
