

// init socket io
var socket;

// setup p5 canvas
function setup() {
    createCanvas(1280,720);
    background(50);

    // connect to server via socket
    socket = io.connect('http://dicklabyvr.duckdns.org:3000');

    // recieve mouse data and instance drawing
    socket.on('objectData', newDrawing)
}

// draw new drawing using recieved data
function newDrawing(data) {

    let boxSize = 88;

    rectMode(CORNER);
    noStroke();
    fill(50, 60);
    rect(0, 0, width, height);

    noStroke();
    fill(255, 0, 128);
    ellipse(data.x, data.y, 8, 8);


    rectMode(CENTER);
    strokeWeight(4);
    stroke(255, 0, 128);
    noFill()
    rect(data.x, data.y, boxSize, boxSize, 8);



    noStroke();
    fill(255, 0, 128);
    textSize(18);
    text(data.id, data.x - (boxSize / 2), (data.y - (boxSize/2)) - 13);

}


/* // send mouse data to server
function mouseDragged() {
    console.log('sending: ' + mouseX + ',' + mouseY);

    // create js object
    var data = {
        x: mouseX,
        y: mouseY
    }

    // send the data
    socket.emit('objectData', data);

    // draw on self canvas
    noStroke();
    fill(255);
    ellipse(mouseX, mouseY, 36, 36);

} */


// p5 draw func
function draw() {


}


