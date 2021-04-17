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
    pub fn propagate(&self, mut input: Vec<f32>) -> Vec<f32> {
        todo!();
        //send the input to the next layer, where it processes it, after looping through all layers, return input

        input
    }
}

impl Layer {
    fn propagate(&self, mut input: Vec<f32>) -> Vec<f32> {
        todo!();
        //loops over each neuron in the layer, gives it the input, and expects a f32 from it
    }
}

impl Neuron {
    fn propagate(&self, mut input: Vec<f32>) -> f32 {
        todo!();
        //loops over all the input values, multiplies it by the given weight, and then adds bias
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
