
// express server init
var express = require('express');
var app = express();

// listen for new socket connections
var server = app.listen(3000);

// serve static files
app.use(express.static('public'));

console.log("server running");

// socket io init
var socket = require('socket.io');
var io = socket(server);


// on new connection to server exec function
io.sockets.on('connection', newConnection);


// new connection function
function newConnection(socket) {
    console.log('new connection: ' + socket.id);

    // data recieved
    socket.on('objectData', objectMsg);

    // exec func on data recieved
    function objectMsg(data) {
        //console.log(data);

        // broadcase message to other users
        socket.broadcast.emit('objectData', data);


    }
}