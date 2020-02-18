const submit_btn = document.querySelector('#submit');
const sqft = document.querySelector('#sqft');
const bathrooms = document.querySelector('#bathrooms');
const bedroomsA = document.querySelector('#bedrooms-ag');
const bedroomsB = document.querySelector('#bedrooms-ag');
const parking = document.querySelector('#parking');
const district = document.querySelector('#district');
const type = document.querySelector('#type');
const priceTag = document.querySelector('#price');

// async function getPrediction(json_data) {
//     const response = await fetch('http://127.0.0.1:5000/api/house', {method: 'POST', body: JSON.stringify(json_data)})
// }


submit_btn.addEventListener('click', () => {
    priceTag.innerText = '$1,310,100';
    console.log('displayed price');
    // const data = {'bathrooms': pars  eInt(bathrooms.value),
    //             'sqft': parseInt(sqft.value),
    //             'parking': parseInt(parking.value),
    //             'bedrooms_ag': parseInt(bedroomsA.value),
    //             'bedrooms_bg': parseInt(bedroomsB.value),
    //             'housing_type': type.value,
    //             'district': district.value}
    // console.log(data);
    // console.log(JSON.stringify(data));
    // // getPrediction(data);
    // // fetch("http://127.0.0.1:5000/api/house", {
    // //   method: "POST",
    // //   body: JSON.stringify(data)
    // // })
    // // .then( (response) => {
    // //     console.log(response);
    // //     //do something awesome that makes the world a better place
    // // });
    // const house_data = {'bathrooms': 2, 'sqft': 1337, 'parking': 1, 'bedrooms_ag': 1, 'bedrooms_bg': 1, 'housing_type': 'Condo Apt', 'district': 'Waterfront Communities-The Island'};
    // console.log(JSON.stringify(house_data));
    // fetch('http://127.0.0.1:5000/api/house', {
    //     method: 'POST',
    //     headers: 'Access-Control-Allow-Origin',
    //     body: JSON.stringify(house_data)
    //     }).then(response => {
    //         console.log(response);
    //     });
});