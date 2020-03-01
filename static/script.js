const submit_btn = document.querySelector('#submit');
const sqft = document.querySelector('#sqft');
const bathrooms = document.querySelector('#bathrooms');
const bedroomsA = document.querySelector('#bedrooms-ag');
const bedroomsB = document.querySelector('#bedrooms-ag');
const parking = document.querySelector('#parking');
const district = document.querySelector('#district');
const type = document.querySelector('#type');
const priceTag = document.querySelector('#price-tag');
const finalPrice = document.querySelector('#price');
// add function to validate input and create default values

async function getHousePrediction(data) {
    const response = await fetch('http://127.0.0.1:5000/api/house',
                                    {method: 'POST', body: JSON.stringify(data)});
    const price = await response.json();
    console.log(price);
    console.log(typeof price);
    // priceTag.innerText = 'Predicted Price: ';
    finalPrice.innerText = `$${price}`;

}

submit_btn.addEventListener('click', () => {
    const data = {'bathrooms': parseInt(bathrooms.value),
                'sqft': parseInt(sqft.value),
                'parking': parseInt(parking.value),
                'bedrooms_ag': parseInt(bedroomsA.value),
                'bedrooms_bg': parseInt(bedroomsB.value),
                'housing_type': type.value,
                'district': district.value};

    getHousePrediction(data);

    // priceTag.innerText = predictedPrice;

    // const data = {'bathrooms': 2, 'sqft': 2555, 'parking': 0, 'bedrooms_ag': 1, 'bedrooms_bg': 1, 'housing_type': 'Condo Apt', 'district': 'Waterfront Communities-The Island'}

    // console.log(JSON.stringify(data));
    // console.log('sending data to api...');
    // fetch('http://127.0.0.1:5000/api/house',
    //     {method: 'POST',
    //         body: JSON.stringify(data)
    //     }).then(response => console.log(response.json());
    // // fetch('http://127.0.0.1:5000/api/house', {
    // //     method: 'POST',
    // //     headers: 'Access-Control-Allow-Origin',
    // //     body: JSON.stringify(data)
    // //     }).then(response => {
    // //         console.log(response);
    // //     });
    // // console.log(JSON.stringify(data));

});