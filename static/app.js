const submit_btn = document.querySelector('#submit');
const priceTag = document.querySelector('#price');

submit_btn.addEventListener('click', () => {
    console.log('changing...');
    priceTag.innerText = '$1,353,040';
    console.log('changed inner text')
});