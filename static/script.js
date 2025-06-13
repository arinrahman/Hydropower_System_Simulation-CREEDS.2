$(document).ready(function () {
    $('#dark-mode-toggle').click(function () {
        $.post('/toggle-dark-mode', function (data) {
            $('body').toggleClass('dark-mode');
        });
    });
});


$(document).ready(function () {
    function updateRangeInputValue(inputId) {
        var value = $('#' + inputId).val();
        $('#' + inputId + '-value').text(value);
    }

    // Update range input values when user interacts with them
    $('#temperature, #release, #inflow, #solar').on('input', function () {
        updateRangeInputValue($(this).attr('id'));
    });
    
    $('#controls-form input[type=range]').change(function () {
        submitForm();
    });

    submitForm();
});

$(document).ready(function () {
    var years = parseInt(localStorage.getItem('years')) || 0;
    var days = parseInt(localStorage.getItem('days')) || 0;
    var hours = parseInt(localStorage.getItem('hours')) || 0;

    function updateTime() {
        days++;
        if (days >= 365) {
            years++;
            days = 0;
        }

        $('.time #time-value').text(years + ' years, ' + days + ' days');

        // Store updated values in localStorage
        localStorage.setItem('years', years);
        localStorage.setItem('days', days);
        localStorage.setItem('hours', hours);

        submitForm();
    }

    function updateEnergyOutput() {
        $.ajax({
            type: 'GET',
            url: '/get_energy_output',
            success: function (data) {
                var formattedOutput = parseFloat(data.energy_output).toFixed(2);
                var solaroutput = parseFloat(data.solar_output).toFixed(2);
                var hydroOutput = parseFloat(data.hydro_output).toFixed(2);
            
                $('.energy-output #energy-output-value').text(formattedOutput);
                $('.energy-output #solar-output-value').text(solaroutput);
                $('.energy-output #hydro-output-value').text(hydroOutput);
                updateCharts();
                updateChart(); 
            },
            error: function (xhr, status, error) {
                console.error(error);
            }
        });
    }
    setInterval(function () {
        updateEnergyOutput();
        updateTime();
    }, 1000);
});

function submitForm() {
    $.ajax({
        type: 'POST',
        url: '/update',
        data: $('#controls-form').serialize(),
        success: function (data) {
            var waterLevel = data.water_level;
            $('#water').css('height', waterLevel + '%');
        },
        error: function (xhr, status, error) {
            console.error(error);
        }
    });
}

function updateChart() {
    $.ajax({
        type: 'GET',
        url: '/get_water_level_data',
        success: function (data) {
            var labels = [];
            var values = [];

            data.forEach(function (entry) {
                labels.push(entry.time);
                values.push(entry.water_level);
            });

            myChart.data.labels = labels;
            myChart.data.datasets[0].data = values;
            myChart.update();
        },
        error: function (xhr, status, error) {
            console.error(error);
        }
    });
}

function updateCharts(){
    $.ajax({
        type: 'GET',
        url: '/get_energy_output',
        success: function (data) {
            var labels = [];
            var values = [];

            data.general_power_out.forEach(function (entry) {
                labels.push(entry.time);
                values.push(entry.energy_output);
            });

            myChart1.data.labels = labels;
            myChart1.data.datasets[0].data = values;
            myChart1.update();


            var labels = [];
            var values = [];

            data.hydro_power_out.forEach(function (entry) {
                labels.push(entry.time);
                values.push(entry.hydro_power);
            });

            myChart2.data.labels = labels;
            myChart2.data.datasets[0].data = values;
            myChart2.update();

            var labels = [];
            var values = [];

            data.solar_power_out.forEach(function (entry) {
                labels.push(entry.time);
                values.push(entry.solar_power);
            });

            myChart3.data.labels = labels;
            myChart3.data.datasets[0].data = values;
            myChart3.update();

        },
        error: function (xhr, status, error) {
            console.error(error);
        }
    });
}
