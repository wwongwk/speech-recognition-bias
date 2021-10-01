var rec

$(document).ready(function () {
    var timeInterval = NaN
    var count = 0
    var txt_start = "start", txt_stop = "stop"

    onload()

    $('button[value="trigger"]').on('click', function () {
        if (!rec){
            showToast("Please check your microphone permission")
            return
        }
        if ($(this).text() == txt_start) {
            timeInterval = setInterval(function () {
                count++;
                $('h1[value="time"]').text(count + " s")
            }, 1000)
            $(this).text(txt_stop)
            rec.start()
        } else if ($(this).text() == txt_stop) {
            clearInterval(timeInterval)
            count = 0
            $('h1[value="time"]').text(count + " s")
	    $('button[value="trigger"]').prop('disabled', true)
            $(this).text(txt_start)
            rec.stop()
        } else {
            // do nothing
        }
    });
    $('button[value="trigger"]').text(txt_start)
});
var toastTimer
function showToast(text) {
    if (!toastTimer){
        clearTimeout(toastTimer)
    }
    $('span[class=toast]').text(text)
    $('span[class=toast]').show()
    toastTimer = setTimeout(() => {
        $('span[class=toast]').hide()
    }, 2000);
}

function postdata(data) {
    var formData = new FormData();
    formData.append("audio", data, "audio.ogg")
    $.post({
        url: "/ad",
        data: formData,    
        processData: false,
        contentType: false,
        success:function(e){
	    $('button[value="trigger"]').prop('disabled', false)
            $('p[value="result"]').text(e)
        },
        error:function(e) {
	    $('button[value="trigger"]').prop('disabled', false)
            showToast("Failed")
        }
    });
}

function onload() {
    if (navigator.mediaDevices === undefined) {
        showToast("Your browser is not supported")
	return
    }
    navigator.mediaDevices.getUserMedia({ video: false, audio: true }).then(stream => {
        window.localStream = stream; 
        rec = new MediaRecorder(stream, { mineType: "audio/mp3" });
        console.error(rec)
        rec.ondataavailable = e => {
	    if (e.data === undefined) {
	        return
	    }
            postdata(e.data)
        };
    }).catch(err => {
        console.log("u got an error:" + err)
    });
}
